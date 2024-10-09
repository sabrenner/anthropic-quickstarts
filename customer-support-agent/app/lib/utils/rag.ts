import tracer from 'dd-trace';
const llmobs = tracer.llmobs;

import {
  BedrockAgentRuntimeClient,
  KnowledgeBaseRetrievalResult,
  RetrieveCommand,
  RetrieveCommandInput,
} from "@aws-sdk/client-bedrock-agent-runtime";
import { RAGSource } from '@/app/lib/utils/types';

console.log("🔑 Have AWS AccessKey?", !!process.env.BAWS_ACCESS_KEY_ID);
console.log("🔑 Have AWS Secret?", !!process.env.BAWS_SECRET_ACCESS_KEY);

const bedrockClient = new BedrockAgentRuntimeClient({
  region: "us-east-1", // Make sure this matches your Bedrock region
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
    sessionToken: process.env.AWS_SESSION_TOKEN!,
  },
});

async function retrieveContext(
  query: string,
  knowledgeBaseId: string,
  n: number = 3,
): Promise<{
  context: string;
  isRagWorking: boolean;
  ragSources: RAGSource[];
}> {
  try {
    if (!knowledgeBaseId) {
      console.error("knowledgeBaseId is not provided");
      return {
        context: "",
        isRagWorking: false,
        ragSources: [],
      };
    }

    const input: RetrieveCommandInput = {
      knowledgeBaseId: knowledgeBaseId,
      retrievalQuery: { text: query },
      retrievalConfiguration: {
        vectorSearchConfiguration: { numberOfResults: n },
      },
    };

    const command = new RetrieveCommand(input);
    const response = await llmobs.trace('tool', { name: 'fetchDocuments' }, async () => {
      llmobs.annotate({ inputData: { query }});
      const response = await bedrockClient.send(command);

      if (response.retrievalResults) {
        llmobs.annotate({
          outputData: response.retrievalResults
        });
      }

      return response
    })

    // Parse results
    const rawResults = response?.retrievalResults || [];
    const ragSources = llmobs.trace('task', { name: 'parseResults' }, () => {
      llmobs.annotate({ inputData: rawResults })
      const ragSources: RAGSource[] = rawResults
        .filter((res: any) => res.content && res.content.text)
        .map((result: any, index: number) => {
          const uri = result?.location?.s3Location?.uri || "";
          const fileName = uri.split("/").pop() || `Source-${index}.txt`;

          return {
            id:
              result.metadata?.["x-amz-bedrock-kb-chunk-id"] || `chunk-${index}`,
            fileName: fileName.replace(/_/g, " ").replace(".txt", ""),
            snippet: result.content?.text || "",
            score: result.score || 0,
          };
        })
        .slice(0, 1);

      llmobs.annotate({ outputData: ragSources })

      console.log("🔍 Parsed RAG Sources:", ragSources); // Debug log

      return ragSources
    })

    const context = rawResults
      .filter((res: any) => res.content && res.content.text)
      .map((res: any) => res.content.text)
      .join("\n\n");

    llmobs.annotate({
      outputData: ragSources.map(source => ({
        id: source.id,
        name: source.fileName,
        score: source.score,
        text: source.snippet
      }))
    })

    return {
      context,
      isRagWorking: true,
      ragSources,
    };
  } catch (error) {
    console.error("RAG Error:", error);
    return { context: "", isRagWorking: false, ragSources: [] };
  }
}

const LLMObsRetrieveContext = llmobs.wrap('retrieval', retrieveContext);
export { LLMObsRetrieveContext as retrieveContext };
