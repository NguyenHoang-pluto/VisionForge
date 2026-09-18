import type { Metadata, Viewport } from "next";
import "./globals.css";

import { PreferencesProvider } from "@/lib/preferences-provider";
import { QueryProvider } from "@/lib/query-provider";
import { PREFERENCES_BOOTSTRAP } from "@/stores/preferences-store";

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
  // Both themes are real, and which one is in force is a stored preference
  // rather than a media query, so the browser chrome is told at runtime by the
  // bootstrap script rather than declared here.
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    /*
     * `suppressHydrationWarning` covers exactly one thing: the theme, accent
     * and density attributes that the inline script below writes onto this
     * element before React is loaded. React would otherwise report the
     * server's markup (which has none of them) as a mismatch.
     */
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Runs before first paint, so the first frame is already in the
            user's theme. See PREFERENCES_BOOTSTRAP for why. */}
        <script dangerouslySetInnerHTML={{ __html: PREFERENCES_BOOTSTRAP }} />
      </head>
      <body className="h-full bg-ground text-fg antialiased">
        <QueryProvider>
          <PreferencesProvider>{children}</PreferencesProvider>
        </QueryProvider>
      </body>
    </html>
  );
}
