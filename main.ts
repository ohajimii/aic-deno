// main.ts
// Deno Deploy-ready server that exposes:
// - GET /v1/models
// - POST /v1/chat/completions (streaming & non-streaming)
// - serves / (static index.html)
// Config via environment variables (see below).

import { serve, type ServeHandlerInfo } from "https://deno.land/std@0.201.0/http/server.ts";

const JWT_URL = Deno.env.get("JWT_URL") ?? "https://beta.aiipo.jp/apmng/chat/get_jwt.php";
const CHAT_URL = Deno.env.get("CHAT_URL") ?? "https://x162-43-21-174.static.xvps.ne.jp/chat";
const PHPSESSID = Deno.env.get("PHPSESSID") ?? ""; // 必须设置为部署机密
const LOG_SOURCE = Deno.env.get("LOG_SOURCE");
const LOG_KEY = Deno.env.get("LOG_KEY");
const IP_HEADER = Deno.env.get("IP_HEADER"); // Header for client IP, e.g., 'cf-connecting-ip'
const CACHE_KEY = ["cache_jwt"];
const JWT_CACHE_TTL_SECONDS = 9 * 60; // 9 minutes
const USER_AGENT =
  "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Mobile Safari/537.36";
const REFERER =
  "https://beta.aiipo.jp/apmng/chat/llm_chat.php?chat_id=-1&p=0";

// 可选模型列表（来自你给的例子）
const AVAILABLE_MODELS = [
  "gemini-2.5-flash-lite-preview-06-17",
  "gemini-2.5-flash",
  "gemini-2.5-pro",
  "gpt-4.1-nano-2025-04-14",
  "gpt-4.1-mini-2025-04-14",
  "gpt-4.1-2025-04-14",
  "gpt-4o-2024-11-20",
  "o4-mini-2025-04-16"
];

const reasoningEffortMap: Record<string, number> = {
  minimal: 512,
  low: 2048,
  medium: 8192,
  high: 24576,
};

async function getKv() {
  // open Kv (Deno Deploy supports Deno.openKv)
  return await Deno.openKv();
}

async function getCachedJwt(): Promise<string | null> {
  const kv = await getKv();
  const res = await kv.get(CACHE_KEY);
  if (!res.value) return null;
  const { jwt, exp } = res.value as { jwt: string; exp: number };
  const now = Math.floor(Date.now() / 1000);
  if (exp <= now) {
    // expired
    await kv.delete(CACHE_KEY);
    return null;
  }
  return jwt;
}

async function cacheJwt(jwt: string) {
  const kv = await getKv();
  const now = Math.floor(Date.now() / 1000);
  const exp = now + JWT_CACHE_TTL_SECONDS; // cache for 9 minutes
  await kv.set(CACHE_KEY, { jwt, exp });
}

async function fetchJwtFromSource(): Promise<string> {
  // call GET JWT endpoint with PHPSESSID cookie
  const headers = new Headers();
  if (PHPSESSID) {
    headers.set("Cookie", `PHPSESSID=${PHPSESSID}`);
  }
  headers.set("User-Agent", USER_AGENT);
  headers.set("Referer", REFERER);
  const resp = await fetch(JWT_URL, { method: "GET", headers });
  if (!resp.ok) {
    throw new Error(`get_jwt failed: ${resp.status} ${await resp.text()}`);
  }
  const j = await resp.json();
  if (!j?.jwt) throw new Error("no jwt in response");
  await cacheJwt(j.jwt);
  return j.jwt;
}

async function getJwt(): Promise<string> {
  const cached = await getCachedJwt();
  if (cached) return cached;
  return await fetchJwtFromSource();
}

function buildModelsResponse() {
  const now = Math.floor(Date.now() / 1000);
  const data = AVAILABLE_MODELS.map((id, idx) => ({
    id,
    object: "model",
    created: now,
    owned_by:"openai",
  }));
  return {
    object: "list",
    data,
  };
}

function pickQueryAndHistory(messages: any[] = []) {
  // We will take the last message with role === 'user' as current query.
  // The history array will be all messages excluding that last user message.
  if (!Array.isArray(messages) || messages.length === 0) {
    return { query: "", history: [] };
  }
  let lastUserIndex = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === "user") {
      lastUserIndex = i;
      break;
    }
  }
  if (lastUserIndex === -1) {
    // no user message, treat everything as history
    return { query: "", history: messages };
  }
  const query = messages[lastUserIndex].content ?? "";
  const history = messages.slice(0, lastUserIndex);
  return { query, history };
}

async function handleModels(req: Request) {
  return new Response(JSON.stringify(buildModelsResponse()), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

function nowSecs() {
  return Math.floor(Date.now() / 1000);
}

function openAiNonStreamResponse(params: {
  id: string;
  model: string;
  text: string;
}) {
  const { id, model, text } = params;
  return {
    id,
    object: "chat.completion",
    created: nowSecs(),
    model,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: text,
          refusal: null,
        },
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
      completion_tokens_details: {},
    },
    system_fingerprint: "fp_1",
  };
}

function makeChunkObject(params: {
  id: string;
  model: string;
  fragment?: string | null;
  finish?: boolean;
}) {
  const created = nowSecs();
  const base: any = {
    id: params.id,
    object: "chat.completion.chunk",
    created,
    model: params.model,
    choices: [
      {
        index: 0,
        delta: {},
        finish_reason: null,
      },
    ],
  };
  if (params.fragment) {
    base.choices[0].delta = { role: "assistant", content: params.fragment };
  }
  if (params.finish) {
    base.choices[0].finish_reason = "stop";
  }
  return base;
}

function sampleString(str: string, maxLength: number): string {
  if (typeof str !== "string") return "";
  return str.slice(0, maxLength);
}

async function handleChatCompletions(req: Request, info: ServeHandlerInfo) {
  // Only accept POST
  if (req.method !== "POST") return new Response("method not allowed", { status: 405 });

  let body: any;
  try {
    body = await req.json();
  } catch {
    body = {};
  }

  const model = body.model ?? "gemini-2.5-flash-lite-preview-06-17";
  const stream = !!body.stream;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const { query, history } = pickQueryAndHistory(messages);

  // Build payload for source
  const inputs: any = {
    llm_model: model,
    web_search: "off",
    thinking_budget: undefined,
  };

  if (body.thinking_budget !== undefined) {
    const tb = body.thinking_budget;
    if (typeof tb === "number") {
      inputs.thinking_budget = tb;
    } else if (typeof tb === "string") {
      inputs.thinking_budget = reasoningEffortMap[tb] ?? tb;
    } else {
      inputs.thinking_budget = tb;
    }
  } else if (body.inputs?.thinking_budget !== undefined) {
    const tb = body.inputs.thinking_budget;
    if (typeof tb === "number") {
      inputs.thinking_budget = tb;
    } else if (typeof tb === "string") {
      inputs.thinking_budget = reasoningEffortMap[tb] ?? tb;
    } else {
      inputs.thinking_budget = tb;
    }
  }

  // messages for the source: we'll forward the "history" as given by pickQueryAndHistory
  // The source example included fields like id, chat_id, token, llm_model, timestamps; we keep a minimal compatible shape.
  const srcMessages = history.map((m: any, idx: number) => ({
    id: m.id ?? idx,
    role: m.role,
    content: m.content,
    token: m.token ?? 1,
    llm_model: m.llm_model ?? model,
    created_at: m.created_at ?? new Date().toISOString(),
    updated_at: m.updated_at ?? new Date().toISOString(),
    deleted_at: m.deleted_at ?? null,
  }));

  const srcBody = {
    messages: srcMessages,
    query,
    conversation_id: body.conversation_id ?? "",
    user:  "web-user",
    inputs,
    response_mode:  "streaming",
  };
  
  // --- Logging ---
  if (LOG_SOURCE && LOG_KEY) {
    try {
      const sampledBackendRequestBody = {
        ...srcBody,
        query: sampleString(srcBody.query, 256),
        messages: srcBody.messages.map((msg: any) => ({
          role:msg.role,
          content: sampleString(msg.content, 64),
        })),
      };

      const ipHeaderValue = IP_HEADER ? req.headers.get(IP_HEADER) : null;
      const clientIp = ipHeaderValue ?? info.remoteAddr.hostname;

      const logData = {
        clientIp,
        ua: req.headers.get("user-agent"),
        backendRequestBody: sampledBackendRequestBody,
      };

      // Fire-and-forget logging
      fetch(`https://api.logflare.app/logs/json?source=${LOG_SOURCE}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json; charset=utf-8',
          'X-API-KEY': LOG_KEY,
        },
        body: JSON.stringify(logData),
      }).catch(e => console.error('Logging fetch failed:', e));
    } catch (e) {
      console.error('Logging failed:', e);
    }
  }

  const jwt = await getJwt();

  const headers = new Headers();
  headers.set("Content-Type", "application/json");
  headers.set("Authorization", `Bearer ${jwt}`);
  headers.set("Accept", "*/*"); 
  headers.set("Origin", "https://beta.aiipo.jp");
  headers.set("User-Agent", USER_AGENT);
  headers.set("Referer", REFERER);
  // Forward other optional headers if needed
  // Send to source
  const srcResp = await fetch(CHAT_URL, {
    method: "POST",
    headers,
    body: JSON.stringify(srcBody),
  });

  if (!srcResp.ok) {
    const text = await srcResp.text();
    return new Response(JSON.stringify({ error: `source error: ${srcResp.status} ${text}` }), {
      status: 502,
      headers: { "content-type": "application/json" },
    });
  }

  // Non-stream handling: try to parse JSON; if it's SSE/chunks, parse them and aggregate.
  if (!stream) {
    // Try to get JSON directly:
    const contentType = srcResp.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      try {
        const j = await srcResp.json();
        // Try to extract a sensible text from returned structure:
        // If j contains message or data fields, try to extract. Fallback to stringify.
        let text = "";
        if (typeof j.text === "string") text = j.text;
        else if (j?.choices?.[0]?.message?.content) text = j.choices[0].message.content;
        else if (j?.data) text = JSON.stringify(j.data);
        else text = JSON.stringify(j);
        const id = `chatcmpl-${Date.now()}`;
        return new Response(JSON.stringify(openAiNonStreamResponse({ id, model, text })), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      } catch (e) {
        // fallback to stream-like parsing below
      }
    }

    // If not JSON, assume source is streaming/chunked SSE-like. We will parse deltas and accumulate.
    const reader = srcResp.body?.getReader();
    if (!reader) {
      return new Response(JSON.stringify(openAiNonStreamResponse({ id: `chatcmpl-${Date.now()}`, model, text: "" })), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    const decoder = new TextDecoder();
    let buf = "";
    let acc = "";
    while (true) {
      const rr = await reader.read();
      if (rr.done) break;
      buf += decoder.decode(rr.value, { stream: true });
      // strip chunk-size lines if present: many chunked responses include hex-size lines; ignore standalone numeric lines.
      // We'll parse "data: " lines terminated by two newlines.
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const chunk = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 2);
        // remove leading numeric hex length lines
        const lines = chunk.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
        let dataPayloads: string[] = [];
        for (const line of lines) {
          // ignore pure hex length lines
          if (/^[0-9a-fA-F]+$/.test(line)) continue;
          if (line.startsWith("data:")) {
            dataPayloads.push(line.slice(5).trim());
          } else {
            // maybe bare json line
            dataPayloads.push(line);
          }
        }
        for (const payload of dataPayloads) {
          if (payload === "[DONE]") {
            // done
            // nothing else to do
          } else {
            try {
              const pj = JSON.parse(payload);
              if (pj?.event === "message_delta") {
                const frag = pj?.data?.delta ?? "";
                acc += frag;
              } else if (pj?.event === "message") {
                // final message metadata
                const finalText = pj?.data?.text ?? "";
                acc += finalText;
              } else if (pj?.event === "message_end") {
                // finished
              } else {
                // unknown, try to pick text
                if (pj?.data?.delta) acc += pj.data.delta;
              }
            } catch {
              // not JSON, append raw
              acc += payload;
            }
          }
        }
      }
    }
    const id = `chatcmpl-${Date.now()}`;
    return new Response(JSON.stringify(openAiNonStreamResponse({ id, model, text: acc })), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }

  // STREAM mode: we need to translate source SSE-like stream into OpenAI-compatible text/event-stream
  // We'll create a ReadableStream that reads from srcResp.body, parses SSE data: lines, and enqueues OpenAI chunks.
  const srcReader = srcResp.body?.getReader();
  if (!srcReader) {
    return new Response("source returned no body reader", { status: 500 });
  }

  const streamResp = new ReadableStream({
    async start(controller) {
      const decoder = new TextDecoder();
      let buf = "";
      const messageId = `chatcmpl-${Date.now()}`;
      try {
        while (true) {
          const { value, done } = await srcReader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          // parse events terminated by \n\n
          let idx;
          while ((idx = buf.indexOf("\n\n")) !== -1) {
            const block = buf.slice(0, idx).trim();
            buf = buf.slice(idx + 2);
            if (!block) continue;
            const lines = block.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
            // collect data: lines
            let dataParts: string[] = [];
            for (const line of lines) {
              if (/^[0-9a-fA-F]+$/.test(line)) continue; // ignore chunk-size
              if (line.startsWith("data:")) {
                dataParts.push(line.slice(5).trim());
              } else {
                dataParts.push(line);
              }
            }
            const payload = dataParts.join("\n");
            if (!payload) continue;
            if (payload === "[DONE]") {
              // pass final DONE
              controller.enqueue(encoder(`data: [DONE]\n\n`));
              controller.close();
              return;
            }
            let pj: any = null;
            try {
              pj = JSON.parse(payload);
            } catch {
              // not JSON -> forward as raw chunk
              const chunkObj = makeChunkObject({ id: messageId, model, fragment: payload });
              controller.enqueue(encoder(`data: ${JSON.stringify(chunkObj)}\n\n`));
              continue;
            }
            // pj has event types: message_delta, message_end, message, etc
            const ev = pj?.event;
            if (ev === "message_delta") {
              const frag = pj?.data?.delta ?? "";
              const chunkObj = makeChunkObject({ id: messageId, model, fragment: frag });
              controller.enqueue(encoder(`data: ${JSON.stringify(chunkObj)}\n\n`));
            } else if (ev === "message") {
              // maybe contains whole message
              const text = pj?.data?.text ?? "";
              const chunkObj = makeChunkObject({ id: messageId, model, fragment: text });
              controller.enqueue(encoder(`data: ${JSON.stringify(chunkObj)}\n\n`));
            } else if (ev === "message_end") {
              // send a finish chunk
              const chunkObj = makeChunkObject({ id: messageId, model, finish: true });
              controller.enqueue(encoder(`data: ${JSON.stringify(chunkObj)}\n\n`));
              controller.enqueue(encoder(`data: [DONE]\n\n`));
              controller.close();
              return;
            } else {
              // unknown event -> try to find text in pj.data.delta
              if (pj?.data?.delta) {
                const chunkObj = makeChunkObject({ id: messageId, model, fragment: pj.data.delta });
                controller.enqueue(encoder(`data: ${JSON.stringify(chunkObj)}\n\n`));
              } else {
                // forward entire object as a chunk
                const chunkObj = makeChunkObject({ id: messageId, model, fragment: JSON.stringify(pj) });
                controller.enqueue(encoder(`data: ${JSON.stringify(chunkObj)}\n\n`));
              }
            }
          }
        } // read loop
        // if we end without explicit [DONE], send DONE
        controller.enqueue(encoder(`data: [DONE]\n\n`));
        controller.close();
      } catch (err) {
        console.error("stream error", err);
        try {
          controller.enqueue(encoder(`data: [DONE]\n\n`));
        } catch {}
        controller.close();
      } finally {
        try { srcReader.releaseLock(); } catch {}
      }
    },
    pull() {},
    cancel() {
      try { srcReader.cancel(); } catch {}
    },
  });

  const encoder = (s: string) => new TextEncoder().encode(s);

  return new Response(streamResp, {
    status: 200,
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache",
      "connection": "keep-alive",
      "transfer-encoding": "chunked",
    },
  });
}

async function handleStatic(req: Request) {
  const url = new URL(req.url);
  // serve index.html for root
  if (url.pathname === "/" || url.pathname === "/index.html") {
    try {
      const data = await Deno.readTextFile("index.html");
      return new Response(data, { status: 200, headers: { "content-type": "text/html; charset=utf-8" } });
    } catch (e) {
      return new Response("index not found", { status: 404 });
    }
  }
  return new Response("not found", { status: 404 });
}

serve(async (req, info) => {
  const url = new URL(req.url);
  if (url.pathname === "/v1/models" && req.method === "GET") {
    return await handleModels(req);
  }
  if (url.pathname === "/v1/chat/completions" && req.method === "POST") {
    return await handleChatCompletions(req, info);
  }
  // static
  if (url.pathname === "/" || url.pathname.startsWith("/index.html")) {
    return await handleStatic(req);
  }
  return new Response("not found", { status: 404 });
});
