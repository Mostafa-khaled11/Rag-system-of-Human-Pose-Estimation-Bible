export type Health = {status:'ready'|'degraded'; ollama:{ok:boolean;detail:string}; qdrant:{ok:boolean;detail:string}; index:{ok:boolean;detail:string}}
export type Citation = {page:number;chapter:string|null;chunk_id:string;score:number;excerpt:string}
export type Answer = {status:'grounded'|'insufficient_context';answer:string;citations:Citation[];timing:{total_ms:number}}
const base = import.meta.env.VITE_API_BASE_URL ?? ''
async function request<T>(path:string, options?:RequestInit):Promise<T>{const response=await fetch(`${base}${path}`,{...options,headers:{'Content-Type':'application/json',...options?.headers}});if(!response.ok){const data=await response.json().catch(()=>({detail:response.statusText}));throw new Error(data.detail??'Request failed')}return response.json()}
export const api={health:()=>request<Health>('/health'),document:()=>request<{indexed:boolean;metadata:Record<string,unknown>|null}>('/api/documents'),config:()=>request<Record<string,unknown>>('/api/config'),ingest:(force:boolean,chunk_size:number,chunk_overlap:number)=>request('/api/ingest',{method:'POST',body:JSON.stringify({force,chunk_size,chunk_overlap})}),query:(question:string,top_k:number)=>request<Answer>('/api/query',{method:'POST',body:JSON.stringify({question,top_k})})}

