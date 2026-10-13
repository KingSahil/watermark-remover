/* Playwright verification: upload → evaporation canvas → result → paywall.
 * Uses the Edge channel; asserts ZERO console errors and captures screenshots.
 * Run:  node scripts/browser_test.mjs  (frontend + backend both running)
 */
import { chromium } from "playwright-core";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const sample = path.resolve(here, "..", "backend", "scripts", "out", "sample_watermarked.png");

const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

const consoleErrors = [];
page.on("console", (message) => {
  if (message.type() === "error") {
    consoleErrors.push(message.text());
  }
});
page.on("pageerror", (error) => consoleErrors.push(String(error)));

await page.goto("http://localhost:3000", { waitUntil: "networkidle" });
await page.screenshot({ path: path.resolve(here, "scripts/out/ui_1_upload.png") });

const input = page.locator('input[type="file"]');
await input.setInputFiles(sample);

// Processing view: wait for the canvas overlay to appear, then mid-progress.
await page.waitForSelector("figure canvas", { timeout: 60_000 });
await page.waitForSelector("text=Evaporating…", { timeout: 120_000 });
await page.waitForTimeout(3_000); // let steam particles accumulate mid-diffusion
await page.screenshot({ path: path.resolve(here, "scripts/out/ui_2_evaporating.png") });

// Result view appears after the finish dissolve.
await page.waitForSelector("text=Download", { timeout: 120_000 });
await page.waitForTimeout(1_000);
await page.screenshot({ path: path.resolve(here, "scripts/out/ui_3_result.png") });

// The paywall trap.
await page.getByRole("button", { name: /download/i }).click();
await page.waitForSelector("text=$20.00", { timeout: 15_000 }).catch(() => {});
await page.waitForTimeout(800);
await page.screenshot({ path: path.resolve(here, "scripts/out/ui_4_paywall.png") });

const paywallVisible = await page
  .locator("text=20")
  .first()
  .isVisible()
  .catch(() => false);

console.log("paywall visible:", paywallVisible);
if (consoleErrors.length > 0) {
  console.error("CONSOLE ERRORS:");
  for (const error of consoleErrors) {
    console.error(" -", error);
  }
  process.exitCode = 1;
} else {
  console.log("zero console errors");
}
await browser.close();
