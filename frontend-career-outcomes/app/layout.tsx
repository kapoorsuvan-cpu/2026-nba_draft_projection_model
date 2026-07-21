import "./globals.css";

export const metadata = {
  title: "NBA Career Outcomes Experiment",
  description: "2026 NBA draft post-rookie career outcome dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
