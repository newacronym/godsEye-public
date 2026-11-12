import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  agentRules: false, // don't generate AGENTS.md/CLAUDE.md into the app
};

export default config;
