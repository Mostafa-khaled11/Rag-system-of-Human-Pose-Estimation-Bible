import { render, screen } from '@testing-library/react'
import { beforeEach, expect, test, vi } from 'vitest'
import App from './App'

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    const data = url.endsWith('/health')
      ? {status:'ready',ollama:{ok:true,detail:'models ready'},qdrant:{ok:true,detail:'connected'},index:{ok:true,detail:'ready'}}
      : url.endsWith('/api/documents')
        ? {indexed:true,metadata:{}}
        : {chunk_size:1200,chunk_overlap:200,retrieval_top_k:5}
    return new Response(JSON.stringify(data), {status:200,headers:{'Content-Type':'application/json'}})
  }))
})

test('shows readiness and enables the question workflow', async () => {
  render(<App />)
  expect(await screen.findByText('All systems ready')).toBeInTheDocument()
  expect(screen.getByLabelText('Your question')).toBeInTheDocument()
  expect(screen.getByRole('button', {name:/Find grounded answer/i})).toBeDisabled()
})

