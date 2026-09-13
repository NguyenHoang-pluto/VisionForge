import type { Metadata, Viewport } from "next";
import "./globals.css";

import { QueryProvider } from "@/lib/query-provider";

export const metadata: Metadata = {
  title: "VisionForge",
  description: "Video editing workstation with AI-assisted editing.",
};

export const viewport: Viewport = {
  // The editor is a fixed viewport with independently scrolling panels, so the
  // page itself must not zoom or bounce.
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="h-full bg-ground text-fg antialiased">
        <QueryProvider>{children}</QueryProvider>
      </body>
    </html>
  );
}
