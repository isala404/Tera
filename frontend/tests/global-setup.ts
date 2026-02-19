import type { FullConfig } from "@playwright/test";

const API_URL = process.env.VITE_API_URL || "http://localhost:8080";

async function waitForBackend(maxRetries = 30, delayMs = 1000): Promise<void> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch(`${API_URL}/_api/health`);
      if (response.ok) {
        console.log("Backend is ready");
        return;
      }
    } catch {
      // Backend not ready yet
    }
    console.log(`Waiting for backend... (${i + 1}/${maxRetries})`);
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }
  throw new Error("Backend did not become ready in time");
}

export default async function globalSetup(_config: FullConfig) {
  await waitForBackend();
}
