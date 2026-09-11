import type { Metadata } from "next";
import "./globals.css";

import { QueryProvider } from "@/lib/query-provider";

export const metadata: Metadata = {
  title: "VisionForge",
  description: "AI-assisted media creation platform",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-100 antialiased">
        <QueryProvider>{children}</QueryProvider>
      </body>
    </html>
  );
}
