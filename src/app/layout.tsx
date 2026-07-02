import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TitanML - AI Knowledge Nexus",
  description: "An interactive knowledge graph and visual encyclopedia for AI concepts, created by Abhishek Shah.",
  keywords: ["AI", "Machine Learning", "Knowledge Graph", "RAG", "LLM", "Agentic AI", "Abhishek Shah"],
  authors: [{ name: "Abhishek Shah" }],
  icons: {
    icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#x1F9E0;</text></svg>",
  },
  openGraph: {
    title: "TitanML - AI Knowledge Nexus",
    description: "Interactive visual encyclopedia of AI concepts, terms, and relationships.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        {/* eslint-disable-next-line @next/next/no-page-custom-font */}
        <link
          href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
        {/* eslint-disable-next-line @next/next/no-css-tags */}
        <link rel="stylesheet" href="/css/styles.css" />
        {/* eslint-disable-next-line @next/next/no-css-tags */}
        <link rel="stylesheet" href="/css/architectureStyles.css" />
        {/* eslint-disable-next-line @next/next/no-css-tags */}
        <link rel="stylesheet" href="/css/knowledgePath.css" />
      </head>
      <body suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}