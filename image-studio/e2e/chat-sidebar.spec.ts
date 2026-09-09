import { expect, test } from "@playwright/test";

test("chat opens to the left, keeps the workspace usable, and retains its draft", async ({ page }) => {
  await page.goto("/");
  const toggle = page.getByRole("button", { name: "Chat with image assistant", exact: true });
  const sidebar = page.locator("#image-assistant");
  const inspector = page.locator(".inspector");
  await expect(toggle).toHaveAttribute("aria-expanded", "false");
  await expect(sidebar).toBeHidden();
  const closedPosition = (await inspector.boundingBox())!.x;

  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-expanded", "true");
  await expect(sidebar).toBeVisible();
  await expect.poll(async () => (await inspector.boundingBox())!.x).toBeGreaterThan(closedPosition + 300);
  const chatBounds = (await sidebar.boundingBox())!;
  const inspectorBounds = (await inspector.boundingBox())!;
  const galleryBounds = (await page.locator(".work-area").boundingBox())!;
  expect(chatBounds.x + chatBounds.width).toBeLessThanOrEqual(inspectorBounds.x + 1);
  expect(inspectorBounds.x + inspectorBounds.width).toBeLessThanOrEqual(galleryBounds.x + 1);
  await expect(page.locator(".modal-backdrop")).toHaveCount(0);

  await page.getByRole("textbox", { name: "Message", exact: true }).fill("Keep this idea while I adjust the image");
  await page.getByRole("textbox", { name: "Prompt", exact: true }).fill("A garden pavilion");
  await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toHaveValue("A garden pavilion");
  await page.getByRole("button", { name: "Close chat", exact: true }).click();
  await expect(sidebar).toBeHidden();
  await expect.poll(async () => (await inspector.boundingBox())!.x).toBe(closedPosition);
  await toggle.click();
  await expect(page.getByRole("textbox", { name: "Message", exact: true })).toHaveValue("Keep this idea while I adjust the image");
});

for (const width of [760, 390]) {
  test(`chat and image controls remain reachable at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 800 });
    await page.goto("/");
    await page.getByRole("button", { name: "Chat with image assistant", exact: true }).click();
    await expect(page.locator("#image-assistant")).toBeVisible();
    await page.getByRole("textbox", { name: "Message", exact: true }).fill("A small courtyard");
    await page.getByRole("textbox", { name: "Prompt", exact: true }).fill("A sunlit courtyard");
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    await expect(page.locator(".modal-backdrop")).toHaveCount(0);
  });
}
