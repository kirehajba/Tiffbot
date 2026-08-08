import "./globals.css";

export const metadata = {
  title: "NeuroSpeak — Neuroplasticity & Articulation Coach",
  description: "Daily communication training with AI coaching: drills, roleplay, mock interviews, and progress tracking.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
