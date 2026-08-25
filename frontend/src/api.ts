export type ComponentHealth = {ok:boolean;detail:string}
export type Health = {status:'ready'|'degraded';ollama:ComponentHealth;qdrant:ComponentHealth;index:ComponentHealth}
export type Citation = {page:number;chapter:string|null;chunk_id:string;score:number;excerpt:string}
export type Timing = {embedding_ms:number;retrieval_ms:number;reranking_ms:number;generation_ms:number;total_ms:number}
export type Answer = {
  status:'grounded'|'insufficient_context';answer:string;citations:Citation[]
  retrieved_chunks:Array<Citation & {source_filename:string}>;timing:Timing;request_id:string
  answerable:boolean;confidence:number;insufficient_context:boolean;reranking_applied:boolean;reranker_model:string|null
}
type ApiErrorBody = {code:string;message:string;retryable:boolean;request_id:string;details?:unknown}
export type StreamEvent =
  | {type:'status';phase:'retrieving';request_id:string}
  | {type:'retrieval';phase:'generating'|'completed';candidate_count:number;passages:Answer['retrieved_chunks'];timing:Timing;request_id:string}
  | {type:'token';text:string;request_id:string}
  | {type:'final';data:Answer}
  | {type:'done';request_id:string}
  | {type:'error';error:ApiErrorBody}
const base = import.meta.env.VITE_API_BASE_URL ?? ''

export class ApiError extends Error {
  code:string;retryable:boolean;requestId:string
  constructor(error:ApiErrorBody){super(error.message);this.name='ApiError';this.code=error.code;this.retryable=error.retryable;this.requestId=error.request_id}
}

async function parseError(response:Response):Promise<ApiError>{
  const data=await response.json().catch(()=>null) as {error?:ApiErrorBody;detail?:string}|null
  if(data?.error)return new ApiError(data.error)
  return new ApiError({code:'REQUEST_FAILED',message:data?.detail??response.statusText??'Request failed',retryable:response.status>=500,request_id:response.headers.get('X-Request-ID')??''})
}

async function request<T>(path:string, options?:RequestInit):Promise<T>{
  const response=await fetch(`${base}${path}`,{...options,headers:{'Content-Type':'application/json',...options?.headers}})
  if(!response.ok)throw await parseError(response)
  return response.json()
}

async function queryStream(question:string,top_k:number,onEvent:(event:StreamEvent)=>void):Promise<void>{
  const response=await fetch(`${base}/api/query/stream`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,top_k})})
  if(!response.ok)throw await parseError(response)
  if(!response.body)throw new ApiError({code:'STREAM_UNAVAILABLE',message:'The browser could not open the response stream.',retryable:true,request_id:response.headers.get('X-Request-ID')??''})
  const reader=response.body.getReader(),decoder=new TextDecoder()
  let buffer=''
  while(true){
    const {done,value}=await reader.read()
    buffer+=decoder.decode(value,{stream:!done})
    const lines=buffer.split('\n');buffer=lines.pop()??''
    for(const line of lines){
      if(!line.trim())continue
      const event=JSON.parse(line) as StreamEvent
      if(event.type==='error')throw new ApiError(event.error)
      onEvent(event)
    }
    if(done)break
  }
  if(buffer.trim()){
    const event=JSON.parse(buffer) as StreamEvent
    if(event.type==='error')throw new ApiError(event.error)
    onEvent(event)
  }
}

export const api={
  health:()=>request<Health>('/ready'),
  document:()=>request<{indexed:boolean;metadata:Record<string,unknown>|null}>('/api/documents'),
  config:()=>request<Record<string,unknown>>('/api/config'),
  ingest:(force:boolean,chunk_size:number,chunk_overlap:number)=>request('/api/ingest',{method:'POST',body:JSON.stringify({force,chunk_size,chunk_overlap})}),
  query:(question:string,top_k:number)=>request<Answer>('/api/query',{method:'POST',body:JSON.stringify({question,top_k})}),
  queryStream,
}
