import { redactSensitiveText } from "openclaw/plugin-sdk/security-runtime";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { MAX_OTEL_CONTENT_ATTRIBUTE_CHARS } from "./service-content-normalization.js";
import { assignOtelModelContentAttributes } from "./service-genai-content.js";

vi.mock("openclaw/plugin-sdk/security-runtime", async (importOriginal) => {
  const actual = await importOriginal<typeof import("openclaw/plugin-sdk/security-runtime")>();
  return { ...actual, redactSensitiveText: vi.fn(actual.redactSensitiveText) };
});

const CONTENT_KEYS = [
  "gen_ai.input.messages",
  "input.value",
  "openclaw.content.input_messages",
] as const;

function capture(inputMessages: unknown[]): Record<string, string | number | boolean> {
  const attributes: Record<string, string | number | boolean> = {};
  assignOtelModelContentAttributes(attributes, { inputMessages }, true);
  return attributes;
}

function stringAttribute(attributes: Record<string, unknown>, key: string): string {
  const value = attributes[key];
  if (typeof value !== "string") {
    throw new Error(`expected string attribute ${key}`);
  }
  return value;
}

describe("model content capture on large transcripts (#158116)", () => {
  beforeEach(() => {
    vi.mocked(redactSensitiveText).mockClear();
  });

  test("redacts each captured string once per model call", () => {
    // Enough long tool outputs that every attribute walks several truncation budgets.
    const toolOutputs = Array.from(
      { length: 40 },
      (_, index) => `tool output ${index} ${"z".repeat(20_000)}`,
    );
    const attributes = capture([
      { role: "user", content: [{ type: "text", text: "summarize the logs" }] },
      ...toolOutputs.map((text, index) => ({
        role: "toolResult",
        toolCallId: `call-${index}`,
        content: [{ type: "text", text }],
      })),
    ]);

    for (const toolOutput of toolOutputs) {
      const calls = vi
        .mocked(redactSensitiveText)
        .mock.calls.filter(([text]) => text === toolOutput);
      expect(calls).toHaveLength(1);
    }
    // The serialized transcript is past 4x the budget, so it is never redacted whole.
    for (const [text] of vi.mocked(redactSensitiveText).mock.calls) {
      expect(text.length).toBeLessThanOrEqual(4 * MAX_OTEL_CONTENT_ATTRIBUTE_CHARS);
    }
    for (const key of CONTENT_KEYS) {
      expect(stringAttribute(attributes, key).length).toBeLessThanOrEqual(
        MAX_OTEL_CONTENT_ATTRIBUTE_CHARS,
      );
      expect(stringAttribute(attributes, key)).toContain("...(truncated)");
    }
    expect(attributes["input.value"]).toBe(attributes["gen_ai.input.messages"]);
  });

  test("masks secrets in every attribute of an oversized transcript", () => {
    const secret = "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef0123";
    const filler = Array.from({ length: 60 }, (_, index) => ({
      role: index % 2 === 0 ? "user" : "assistant",
      content: [{ type: "text", text: `turn ${index} ${"w".repeat(20_000)}` }],
    }));
    const attributes = capture([
      { role: "user", content: [{ type: "text", text: `deploy with key ${secret}` }] },
      ...filler,
    ]);

    for (const key of CONTENT_KEYS) {
      const value = stringAttribute(attributes, key);
      expect(value.length).toBeLessThanOrEqual(MAX_OTEL_CONTENT_ATTRIBUTE_CHARS);
      expect(value).toContain("deploy with key sk-ant");
      expect(value).not.toContain(secret);
    }
  });

  test("exports a value whole when masking brings it under the attribute budget", () => {
    const token = `sk-${"a".repeat(MAX_OTEL_CONTENT_ATTRIBUTE_CHARS + 10_000)}`;
    const attributes = capture([{ role: "user", content: [{ type: "text", text: token }] }]);

    expect(JSON.parse(stringAttribute(attributes, "gen_ai.input.messages"))).toEqual([
      { role: "user", parts: [{ type: "text", content: "sk-aaa…aaaa" }] },
    ]);
  });

  test("drops image bytes too large to export whole and keeps smaller images", () => {
    const largeImage = "A".repeat(MAX_OTEL_CONTENT_ATTRIBUTE_CHARS + 1);
    const smallImage = "iVBORw0KGgo=";
    const attributes = capture([
      {
        role: "user",
        content: [
          { type: "text", text: "compare these" },
          { type: "image", mimeType: "image/png", data: smallImage },
        ],
      },
      {
        role: "toolResult",
        toolCallId: "call-1",
        content: [
          { type: "text", text: "screenshot" },
          { type: "image", mimeType: "image/png", data: largeImage },
        ],
      },
    ]);

    expect(JSON.parse(stringAttribute(attributes, "gen_ai.input.messages"))).toEqual([
      {
        role: "user",
        parts: [
          { type: "text", content: "compare these" },
          { type: "blob", modality: "image", mime_type: "image/png", content: smallImage },
        ],
      },
      {
        role: "tool",
        parts: [
          {
            type: "tool_call_response",
            id: "call-1",
            response: [
              { type: "text", text: "screenshot" },
              { type: "image", mimeType: "image/png" },
            ],
          },
        ],
      },
    ]);
    const raw = JSON.parse(stringAttribute(attributes, "openclaw.content.input_messages"));
    expect(raw[0].content[1]).toEqual({ type: "image", mimeType: "image/png", data: smallImage });
    expect(raw[1].content[1]).toEqual({ type: "image", mimeType: "image/png" });
  });
});
