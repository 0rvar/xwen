import { expect, test } from "@playwright/test";

test("frontend console messages, uncaught exceptions and rejections reach the log transport", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Gallery", exact: true })).toBeVisible();
  await page.evaluate(() => {
    console.info("logging test info");
    console.warn("logging test warning");
    console.error(new Error("logging test caught error"));
    console.debug({ api_key: "do-not-log-this", image: "data:image/png;base64,AAAA" });
    setTimeout(() => { throw new Error("logging test uncaught exception"); }, 0);
    void Promise.reject(new Error("logging test unhandled rejection"));
  });
  await expect.poll(() => page.evaluate(() => (window as typeof window & { __XWEN_PREVIEW_LOGS__?: { source: string; message: string }[] }).__XWEN_PREVIEW_LOGS__ ?? [])).toEqual(expect.arrayContaining([
    expect.objectContaining({ source: "frontend.console.info", message: expect.stringContaining("logging test info") }),
    expect.objectContaining({ source: "frontend.console.warn", message: expect.stringContaining("logging test warning") }),
    expect.objectContaining({ source: "frontend.console.error", message: expect.stringContaining("logging test caught error") }),
    expect.objectContaining({ source: "frontend.uncaught", message: expect.stringContaining("logging test uncaught exception") }),
    expect.objectContaining({ source: "frontend.unhandledrejection", message: expect.stringContaining("logging test unhandled rejection") }),
  ]));
  const logs = await page.evaluate(() => JSON.stringify((window as typeof window & { __XWEN_PREVIEW_LOGS__?: unknown[] }).__XWEN_PREVIEW_LOGS__));
  expect(logs).not.toContain("do-not-log-this");
  expect(logs).not.toContain("AAAA");
  await page.getByRole("button", { name: "Server settings" }).click();
  await page.getByRole("button", { name: "Show application log" }).click();
  await expect(page.getByRole("alert")).toHaveCount(0);
});
