import { test, expect } from "@playwright/test";

test.describe("Application", () => {
  test("homepage loads successfully", async ({ page }) => {
    await page.goto("/");

    // Page should load without errors
    await expect(page.locator("body")).toBeVisible();

    // Main heading should be present
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  });

  test("no console errors on page load", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        errors.push(msg.text());
      }
    });

    await page.goto("/");
    await page.waitForTimeout(3000);

    const unexpectedErrors = errors.filter(
      (e) =>
        !e.includes("net::ERR") &&
        !e.includes("favicon") &&
        !e.includes("EventSource"),
    );
    expect(unexpectedErrors).toHaveLength(0);
  });

  test("backend health check", async ({ request }) => {
    const response = await request.get("http://localhost:8080/_api/health");
    expect(response.ok()).toBeTruthy();
  });
});
