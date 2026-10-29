import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DeepShield | Explainable Media Forensics Platform (MCA21491)",
  description: "Hybrid EfficientNet-B0 + ViT-B/16 + 2D-FFT Deepfake & Synthetic Face Detection Platform",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#090d16] text-slate-100 antialiased selection:bg-sky-500/30">
        {children}
      </body>
    </html>
  );
}
