import "./globals.css";

export const metadata = {
  title: "NBA College Projection Model",
  description: "2026 NBA draft prospect projection dashboard",
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
