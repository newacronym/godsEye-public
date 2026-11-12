import type { Metadata } from "next";
import { GeistMono } from "geist/font/mono";
import { GeistSans } from "geist/font/sans";
import "./globals.css";
import { Rail } from "@/components/rail";

export const metadata: Metadata = {
  title: "godsEye",
  description: "Ask your cameras anything.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable}`}>
      {/* Browser extensions (ColorZilla, Grammarly) add attributes to <body> before React loads. */}
      <body className="min-h-[100dvh]" suppressHydrationWarning>
        <div className="flex min-h-[100dvh]">
          <Rail />
          <main className="min-w-0 flex-1">{children}</main>
        </div>
      </body>
    </html>
  );
}
