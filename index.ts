import { ChatAnthropic } from "@langchain/anthropic";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import { Document } from "@langchain/core/documents";
import { Annotation } from "@langchain/langgraph";

import { concat } from "@langchain/core/utils/stream";

import { StateGraph } from "@langchain/langgraph";

import * as dotenv from "dotenv";
dotenv.config();

console.log("Hello via Bun!");

const llm = new ChatAnthropic({
  model: "claude-3-5-sonnet-20240620",
  temperature: 0,
});

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-large",
});

const vectorStore = new MemoryVectorStore(embeddings);

// Load and chunk contents of blog
const pTagSelector = "p";
const cheerioLoader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  {
    selector: pTagSelector,
  },
);

const docs = await cheerioLoader.load();

console.assert(docs.length === 1);
console.log(`Total characters: ${docs[0].pageContent.length}`);

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const allSplits = await splitter.splitDocuments(docs);
console.log(`Split blog post into ${allSplits.length} subdocuments`);

// Storing documents

await vectorStore.addDocuments(allSplits);

// RAG

const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");

// Example
const example_prompt = await promptTemplate.invoke({
  context: "(context goes here)",
  question: "(question goes here)",
});
const example_messages = example_prompt.messages;

console.assert(example_messages.length === 1);
console.log(example_messages[0].content);

const InputStateAnnotation = Annotation.Root({
  question: Annotation<string>,
});

const StateAnnotation = Annotation.Root({
  question: Annotation<string>,
  context: Annotation<Document[]>,
  answer: Annotation<string>,
});

const retrieve = async (state: typeof InputStateAnnotation.State) => {
  const retrievedDocs = await vectorStore.similaritySearch(state.question);
  return { context: retrievedDocs };
};

const generate = async (state: typeof StateAnnotation.State) => {
  const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
  const messages = await promptTemplate.invoke({
    question: state.question,
    context: docsContent,
  });

  const response = await llm.invoke(messages);
  return { answer: response.content };
};

const graph = new StateGraph(StateAnnotation)
  .addNode("retrieve", retrieve)
  .addNode("generate", generate)
  .addEdge("__start__", "retrieve")
  .addEdge("retrieve", "generate")
  .addEdge("generate", "__end__")
  .compile();

let inputs = { question: "What is Task Decomposition?" };

console.log("\n====\n");
for await (const chunk of await graph.stream(inputs, {
  streamMode: "updates",
})) {
  console.log(chunk);
  console.log("\n====\n");
}

//
// const result = await graph.invoke(input);
// console.log(result.context.slice(0, 2));
// console.log(`\nAnswer: ${result["answer"]}`);
