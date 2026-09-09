import { describe, expect, test } from "bun:test";
import { formatLogValues, setLogSecrets } from "./logging";

describe("frontend log records", () => {
  test("retains exception details and safely represents circular causes", () => {
    const error = new Error("image read failed");
    error.cause = error;
    const line = formatLogValues([error]);
    expect(line).toContain("image read failed");
    expect(line).toContain("logging.test.ts");
    expect(line).toContain("[circular]");
  });

  test("removes credentials and image payloads from nested logs and errors", () => {
    setLogSecrets("configured-key-123");
    const line = formatLogValues([{ api_key: "unknown-draft-key", nested: { authorization: "another-secret", data_url: "data:image/png;base64,AAAA" } }, new Error("configured-key-123 data:image/png;base64,AABBCC== Bearer yet-another-secret")]);
    for (const secret of ["configured-key-123", "unknown-draft-key", "another-secret", "AABBCC", "AAAA"]) expect(line).not.toContain(secret);
    expect(line).toContain("[redacted]");
    expect(line).toContain("[image data]");
  });

  test("logging broken getters and oversized values cannot break the application", () => {
    const value = { get detail() { throw new Error("broken getter"); } };
    expect(formatLogValues([value])).toContain("[unreadable]");
    expect(formatLogValues(["a".repeat(100_000)]).length).toBeLessThanOrEqual(8_000);
    expect(formatLogValues([1n])).toContain("1");
  });
});
