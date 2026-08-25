import { FormEvent, useEffect, useState } from 'react'
import { api, Answer, ApiError, Health, StreamEvent } from './api'
import './styles.css'

type QueryPhase='idle'|'retrieving'|'generating'|'completed'|'failed'

function errorMessage(error:unknown,fallback:string){
  if(error instanceof ApiError)return `${error.message}${error.requestId?` (Request ${error.requestId})`:''}`
  return error instanceof Error?error.message:fallback
}

export default function App(){
 const [health,setHealth]=useState<Health|null>(null),[indexed,setIndexed]=useState(false),[question,setQuestion]=useState(''),[answer,setAnswer]=useState<Answer|null>(null),[streamText,setStreamText]=useState(''),[error,setError]=useState(''),[busy,setBusy]=useState(false),[phase,setPhase]=useState<QueryPhase>('idle')
 const [chunkSize,setChunkSize]=useState(1200),[overlap,setOverlap]=useState(200),[topK,setTopK]=useState(5),[dirty,setDirty]=useState(false)
 const refresh=async()=>{try{const [h,d,c]=await Promise.all([api.health(),api.document(),api.config()]);setHealth(h);setIndexed(d.indexed);setChunkSize(Number(c.chunk_size));setOverlap(Number(c.chunk_overlap));setTopK(Number(c.retrieval_top_k));setError('')}catch(e){setError(errorMessage(e,'Service unavailable'))}}
 useEffect(()=>{void refresh()},[])
 const ingest=async(force=false)=>{setBusy(true);setError('');try{await api.ingest(force,chunkSize,overlap);setDirty(false);await refresh()}catch(e){setError(errorMessage(e,'Ingestion failed'))}finally{setBusy(false)}}
 const onStreamEvent=(event:StreamEvent)=>{
   if(event.type==='status')setPhase('retrieving')
   if(event.type==='retrieval')setPhase(event.phase)
   if(event.type==='token'){setPhase('generating');setStreamText(text=>text+event.text)}
   if(event.type==='final'){setAnswer(event.data);setStreamText(event.data.answer);setPhase('completed')}
 }
 const submit=async(e:FormEvent)=>{e.preventDefault();if(!question.trim())return;setBusy(true);setError('');setAnswer(null);setStreamText('');setPhase('retrieving');try{await api.queryStream(question,topK,onStreamEvent)}catch(err){setPhase('failed');setError(errorMessage(err,'Query failed'))}finally{setBusy(false)}}
 const statusLabel=phase==='retrieving'?'Retrieving passages…':phase==='generating'?'Generating grounded answer…':phase==='failed'?'Query failed':phase==='completed'?'Completed':''
 return <main>
  <header><div><p className="eyebrow">LOCAL · PRIVATE · GROUNDED</p><h1>Human Pose Estimation<br/><span>Book Assistant</span></h1></div><div className={`status ${health?.status??'degraded'}`}><i/>{health?.status==='ready'?'All systems ready':'Setup required'}</div></header>
  <section className="service-grid" aria-label="Service status">{[['Ollama',health?.ollama],['Qdrant',health?.qdrant],['Book index',health?.index]].map(([name,state])=><article key={name as string}><b>{name as string}</b><span className={state && (state as Health['ollama']).ok?'ok':'bad'}>{state?(state as Health['ollama']).detail:'Checking…'}</span></article>)}</section>
  <div className="layout"><section className="panel ask"><h2>Ask the book</h2><p>Answers use only retrieved passages and include page citations.</p><form onSubmit={submit}><label htmlFor="question">Your question</label><textarea id="question" value={question} maxLength={2000} onChange={e=>setQuestion(e.target.value)} placeholder="How are 2D human poses represented?" rows={4}/><button disabled={busy||!indexed||!question.trim()}>{busy?'Working…':'Find grounded answer'} <span>→</span></button></form>{statusLabel&&(busy||phase==='failed')&&<p className="query-status" role="status">{statusLabel}</p>}{error&&<p role="alert" className="error">{error}</p>}
  {(streamText||answer)&&<article className={`answer ${answer?.insufficient_context?'insufficient':''}`} aria-live="polite"><div className="answer-head"><span>{answer?(answer.insufficient_context?'Insufficient context':'Grounded answer'):statusLabel}</span>{answer&&<small>{Math.round(answer.timing.total_ms)} ms · confidence {Math.round(answer.confidence*100)}%</small>}</div><p>{streamText}<span className={busy?'stream-cursor':'stream-cursor hidden'} aria-hidden="true">▍</span></p>{answer&&answer.citations.length>0&&<div className="citations"><h3>Sources</h3>{answer.citations.map(c=><details key={c.chunk_id}><summary><b>Page {c.page}</b><span>{c.chapter??'Unlabeled section'}</span><em>{c.score.toFixed(3)}</em></summary><p>{c.excerpt}</p></details>)}</div>}</article>}</section>
  <aside className="panel settings"><h2>Index settings</h2><p>Character-based chunk controls. Changes require re-indexing.</p><label>Chunk size <input type="number" min="200" max="8000" value={chunkSize} onChange={e=>{setChunkSize(+e.target.value);setDirty(true)}}/></label><label>Overlap <input type="number" min="0" max="2000" value={overlap} onChange={e=>{setOverlap(+e.target.value);setDirty(true)}}/></label><label>Retrieved passages <input type="number" min="1" max="25" value={topK} onChange={e=>setTopK(+e.target.value)}/></label>{dirty&&<p className="warning">Chunk settings changed. Re-index before querying.</p>}<button className="secondary" disabled={busy||chunkSize<=overlap} onClick={()=>void ingest(indexed)}>{indexed?'Re-index book':'Index book'}</button><small>The book is mounted read-only. Re-indexing replaces the active index only after a successful build.</small></aside></div>
 </main>
}
