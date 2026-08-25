import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, expect, test, vi } from 'vitest'
import App from './App'
import { Answer } from './api'

const grounded:Answer={
  status:'grounded',answer:'Depth is ambiguous [p. 58].',
  citations:[{page:58,chapter:'3D HPE',chunk_id:'c1',score:.91,excerpt:'Multiple 3D poses can have the same projection.'}],
  retrieved_chunks:[],timing:{embedding_ms:4,retrieval_ms:5,reranking_ms:1,generation_ms:90,total_ms:100},
  request_id:'req-1',answerable:true,confidence:.91,insufficient_context:false,reranking_applied:true,reranker_model:'lexical-v1'
}

function ndjson(events:unknown[]){return events.map(event=>JSON.stringify(event)).join('\n')+'\n'}

function installFetch(streamResponse?:()=>Promise<Response>){
  vi.stubGlobal('fetch',vi.fn(async(input:RequestInfo|URL)=>{
    const url=String(input)
    if(url.endsWith('/ready'))return new Response(JSON.stringify({status:'ready',ollama:{ok:true,detail:'models ready'},qdrant:{ok:true,detail:'connected'},index:{ok:true,detail:'ready'}}),{status:200})
    if(url.endsWith('/api/documents'))return new Response(JSON.stringify({indexed:true,metadata:{}}),{status:200})
    if(url.endsWith('/api/config'))return new Response(JSON.stringify({chunk_size:1200,chunk_overlap:200,retrieval_top_k:5}),{status:200})
    if(url.endsWith('/api/query/stream')&&streamResponse)return streamResponse()
    return new Response('{}',{status:200})
  }))
}

async function submit(question='Why is monocular 3D HPE ambiguous?'){
  await screen.findByText('All systems ready')
  fireEvent.change(screen.getByLabelText('Your question'),{target:{value:question}})
  fireEvent.click(screen.getByRole('button',{name:/Find grounded answer/i}))
}

beforeEach(()=>installFetch())

test('shows readiness and enables the question workflow',async()=>{
  render(<App/>)
  expect(await screen.findByText('All systems ready')).toBeInTheDocument()
  expect(screen.getByLabelText('Your question')).toBeInTheDocument()
  expect(screen.getByRole('button',{name:/Find grounded answer/i})).toBeDisabled()
})

test('shows retrieving while a streaming request is pending',async()=>{
  let finish!:(response:Response)=>void
  installFetch(()=>new Promise<Response>(resolve=>{finish=resolve}))
  render(<App/>);await submit()
  expect(screen.getByRole('status')).toHaveTextContent('Retrieving passages')
  expect(screen.getByRole('button',{name:/Working/i})).toBeDisabled()
  finish(new Response(ndjson([{type:'final',data:grounded},{type:'done',request_id:'req-1'}]),{status:200}))
  expect(await screen.findByText('Depth is ambiguous [p. 58].')).toBeInTheDocument()
})

test('renders streaming state, final answer, timing, and citations',async()=>{
  installFetch(async()=>{
    const encoder=new TextEncoder()
    const body=new ReadableStream({start(controller){
      controller.enqueue(encoder.encode(ndjson([{type:'status',phase:'retrieving',request_id:'req-1'},{type:'retrieval',phase:'generating',candidate_count:20,passages:[],timing:grounded.timing,request_id:'req-1'}])))
      setTimeout(()=>{controller.enqueue(encoder.encode(ndjson([{type:'token',text:'Depth is ',request_id:'req-1'},{type:'final',data:grounded},{type:'done',request_id:'req-1'}])));controller.close()},20)
    }})
    return new Response(body,{status:200})
  })
  render(<App/>);await submit()
  expect(await screen.findByText('Generating grounded answer…')).toBeInTheDocument()
  expect(await screen.findByText('Depth is ambiguous [p. 58].')).toBeInTheDocument()
  expect(screen.getByText('Page 58')).toBeInTheDocument()
  expect(screen.getByText(/100 ms · confidence 91%/)).toBeInTheDocument()
})

test('renders an insufficient-context completion without citations',async()=>{
  const insufficient={...grounded,status:'insufficient_context' as const,answer:'The retrieved passages do not contain enough information to answer this question reliably.',citations:[],answerable:false,confidence:0,insufficient_context:true}
  installFetch(async()=>new Response(ndjson([{type:'final',data:insufficient},{type:'done',request_id:'req-1'}]),{status:200}))
  render(<App/>);await submit('Which medicine should I take?')
  expect(await screen.findByText('Insufficient context')).toBeInTheDocument()
  expect(screen.queryByText('Sources')).not.toBeInTheDocument()
})

test('shows structured backend errors with the request ID',async()=>{
  installFetch(async()=>new Response(JSON.stringify({error:{code:'OLLAMA_UNAVAILABLE',message:'The local model is unavailable.',retryable:true,request_id:'req-fail'}}),{status:503,headers:{'Content-Type':'application/json'}}))
  render(<App/>);await submit()
  expect(await screen.findByRole('alert')).toHaveTextContent('The local model is unavailable. (Request req-fail)')
})

test('shows errors delivered after a stream has started',async()=>{
  installFetch(async()=>new Response(ndjson([{type:'status',phase:'retrieving',request_id:'r'},{type:'error',error:{code:'QUERY_TIMEOUT',message:'Answer generation timed out.',retryable:true,request_id:'r'}}]),{status:200}))
  render(<App/>);await submit()
  expect(await screen.findByRole('alert')).toHaveTextContent('Answer generation timed out. (Request r)')
  await waitFor(()=>expect(screen.getByText('Query failed')).toBeInTheDocument())
})
