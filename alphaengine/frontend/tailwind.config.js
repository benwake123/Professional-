/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        dark: { bg: "#0a0a0f", card: "#12121a", border: "#1e1e2e" },
        accent: "#00d4ff",
        profit: "#00ff88",
        loss: "#ff4444",
      },
      fontFamily: { mono: ["JetBrains Mono", "monospace"] },
    },
  },
  plugins: [],
};
