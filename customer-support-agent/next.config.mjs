/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverComponentsExternalPackages: ["@aws-sdk/client-bedrock-agent-runtime"],
  },
  webpack: (config, { isServer, webpack }) => {
    if (isServer) {
      config.externals.push({
        "@aws-sdk/client-bedrock-agent-runtime":
          "commonjs @aws-sdk/client-bedrock-agent-runtime",
      });
    }

    config.resolve.fallback = {
      ...config.resolve.fallback,
      graphql: false
    }

    return config;
  },
};

export default nextConfig;
