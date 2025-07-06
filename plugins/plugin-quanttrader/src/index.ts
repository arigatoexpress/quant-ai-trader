import type { Plugin, Content, HandlerCallback, IAgentRuntime, Memory, State, Provider, ProviderResult } from '@elizaos/core';
import fetch from 'node-fetch';

const fetchSummary = async () => {
  const res = await fetch('http://localhost:5000/api/summary');
  return (await res.json()) as any;
};

const marketProvider: Provider = {
  name: 'QUANT_TRADER_PROVIDER',
  description: 'Provides market summary from Quant AI Trader',
  async get(_runtime: IAgentRuntime, _message: Memory, _state: State | undefined): Promise<ProviderResult> {
    const data = await fetchSummary();
    return { text: 'Market summary retrieved', values: {}, data };
  }
};

const marketAction = {
  name: 'FETCH_MARKET_SUMMARY',
  similes: ['GET_MARKET_DATA'],
  description: 'Fetch trading insights from Quant AI Trader',
  validate: async () => true,
  async handler(_runtime: IAgentRuntime, message: Memory, _state: State | undefined, _options: any, callback?: HandlerCallback) {
    const data = await fetchSummary();
    const text = `${data.outlook}\n${data.highlights.join('\n')}`;
    const content: Content = {
      text,
      actions: ['FETCH_MARKET_SUMMARY'],
      source: message.content.source,
    };
    if (callback) await callback(content);
    return content;
  },
  examples: [] as any,
};

export const quantTraderPlugin: Plugin = {
  name: 'plugin-quanttrader',
  description: 'Integrates Quant AI Trader signals into ElizaOS',
  providers: [marketProvider],
  actions: [marketAction],
};

export default quantTraderPlugin;
