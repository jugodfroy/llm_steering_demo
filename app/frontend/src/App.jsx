import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'
import Presentation from './Presentation'

function App() {
  const [tab, setTab] = useState('presentation') // 'presentation' | 'demo'
  const [vectors, setVectors] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [prompt, setPrompt] = useState('')
  const [strength, setStrength] = useState(8)
  const [layer, setLayer] = useState(15)
  const [maxTokens, setMaxTokens] = useState(200)
  const [baselineText, setBaselineText] = useState('')
  const [promptedText, setPromptedText] = useState('')
  const [steeredText, setSteeredText] = useState('')
  const [generating, setGenerating] = useState(false)
  const [phase, setPhase] = useState(null) // 'baseline' | 'prompted' | 'steered' | 'done'
  const baselineRef = useRef(null)
  const promptedRef = useRef(null)
  const steeredRef = useRef(null)
  const abortRef = useRef(null)

  // Load vectors on mount
  useEffect(() => {
    fetch('/api/vectors')
      .then(r => r.json())
      .then(data => {
        setVectors(data)
        // Auto-select first wow vector
        const first = data.find(v => v.category === 'wow')
        if (first) {
          setSelectedId(first.id)
          setStrength(first.default_strength)
          setLayer(first.layer)
          if (first.example_prompt) setPrompt(first.example_prompt)
        }
      })
      .catch(console.error)
  }, [])

  const selected = vectors.find(v => v.id === selectedId)

  const handleSelectVector = useCallback((vec) => {
    setSelectedId(vec.id)
    setStrength(vec.default_strength)
    setLayer(vec.layer)
    if (vec.example_prompt) setPrompt(vec.example_prompt)
  }, [])

  const handleGenerate = useCallback(async () => {
    if (!selectedId || !prompt.trim() || generating) return

    setGenerating(true)
    setBaselineText('')
    setPromptedText('')
    setSteeredText('')
    setPhase('steered')

    if (abortRef.current) abortRef.current.abort()
    const controller = new AbortController()
    abortRef.current = controller

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          vector_id: selectedId,
          strength,
          layer,
          max_tokens: maxTokens,
        }),
        signal: controller.signal,
      })

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const data = JSON.parse(line.slice(6))

            if (data.type === 'start') {
              setPhase(data.mode)
            } else if (data.type === 'token') {
              if (data.mode === 'baseline') {
                setBaselineText(prev => prev + data.token)
              } else if (data.mode === 'prompted') {
                setPromptedText(prev => prev + data.token)
              } else {
                setSteeredText(prev => prev + data.token)
              }
            } else if (data.type === 'done') {
              setPhase('done')
            }
          } catch {}
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') console.error(err)
    } finally {
      setGenerating(false)
      setPhase('done')
    }
  }, [selectedId, prompt, strength, layer, maxTokens, generating])

  // Auto-scroll
  useEffect(() => {
    if (baselineRef.current) baselineRef.current.scrollTop = baselineRef.current.scrollHeight
  }, [baselineText])
  useEffect(() => {
    if (promptedRef.current) promptedRef.current.scrollTop = promptedRef.current.scrollHeight
  }, [promptedText])
  useEffect(() => {
    if (steeredRef.current) steeredRef.current.scrollTop = steeredRef.current.scrollHeight
  }, [steeredText])

  // Enter to submit
  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleGenerate()
    }
  }, [handleGenerate])

  const wowVectors = vectors.filter(v => v.category === 'wow')
  const ispVectors = vectors.filter(v => v.category === 'isp')
  const otherVectors = vectors.filter(v => v.category === 'other')

  return (
    <>
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <h1>LLM Steering</h1>
          <span className="header-badge">R&D WEEK</span>
        </div>
        <div className="header-tabs">
          <button
            className={`header-tab ${tab === 'presentation' ? 'active' : ''}`}
            onClick={() => setTab('presentation')}
          >
            Presentation
          </button>
          <button
            className={`header-tab ${tab === 'demo' ? 'active' : ''}`}
            onClick={() => setTab('demo')}
          >
            Demo
          </button>
        </div>
        <span className="header-model">Llama 3.1 8B Instruct &middot; H100</span>
      </header>

      {tab === 'presentation' ? (
        <Presentation />
      ) : (
      <div className="app-layout">
        {/* Sidebar */}
        <aside className="sidebar">
          {wowVectors.length > 0 && (
            <div className="sidebar-section">
              <div className="sidebar-section-title">Showcase</div>
              {wowVectors.map(v => (
                <VectorCard key={v.id} vec={v} active={selectedId === v.id} onClick={() => handleSelectVector(v)} />
              ))}
            </div>
          )}
          {ispVectors.length > 0 && (
            <div className="sidebar-section">
              <div className="sidebar-section-title">ISP Related</div>
              {ispVectors.map(v => (
                <VectorCard key={v.id} vec={v} active={selectedId === v.id} onClick={() => handleSelectVector(v)} />
              ))}
            </div>
          )}
          {otherVectors.length > 0 && (
            <div className="sidebar-section">
              <div className="sidebar-section-title">Autres vecteurs</div>
              {otherVectors.map(v => (
                <VectorCard key={v.id} vec={v} active={selectedId === v.id} onClick={() => handleSelectVector(v)} />
              ))}
            </div>
          )}
        </aside>

        {/* Main */}
        <main className="main-panel">
          {/* Prompt bar */}
          <div className="controls-bar">
            <div className="prompt-input-wrapper">
              <label>Prompt</label>
              <input
                className="prompt-input"
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type a prompt and press Enter..."
              />
            </div>
            <button
              className={`generate-btn ${generating ? 'generating' : ''}`}
              onClick={handleGenerate}
              disabled={generating || !selectedId || !prompt.trim()}
            >
              {generating ? 'Generating...' : 'Generate'}
            </button>
          </div>

          {/* Params row */}
          {selected && (
            <div className="params-row">
              <div className="steered-info">
                <span>{selected.emoji}</span>
                <strong>{selected.label}</strong>
                <span>&mdash; {selected.description}</span>
              </div>
              <div className="param-group">
                <span className="param-label">Strength</span>
                <input
                  type="range" className="param-slider"
                  min={-20} max={20} step={0.5}
                  value={strength}
                  onChange={e => setStrength(Number(e.target.value))}
                />
                <span className="param-value">{strength}</span>
              </div>
              <div className="param-group">
                <span className="param-label">Layer</span>
                <input
                  type="range" className="param-slider"
                  min={0} max={31} step={1}
                  value={layer}
                  onChange={e => setLayer(Number(e.target.value))}
                />
                <span className="param-value">{layer}</span>
              </div>
              <div className="param-group">
                <span className="param-label">Tokens</span>
                <input
                  type="range" className="param-slider"
                  min={50} max={500} step={25}
                  value={maxTokens}
                  onChange={e => setMaxTokens(Number(e.target.value))}
                />
                <span className="param-value">{maxTokens}</span>
              </div>
              {generating && (
                <div className={`status-bar ${phase === 'done' ? 'done' : 'generating'}`}>
                  <div className="spinner" />
                  <span>
                    {phase === 'steered' ? 'Generating steered...' :
                     phase === 'baseline' ? 'Generating baseline...' :
                     phase === 'prompted' ? 'Generating prompted...' : 'Done'}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Results */}
          <div className="results-grid">
            {!baselineText && !promptedText && !steeredText && !generating ? (
              <div className="empty-state">
                <div className="empty-state-icon">🧬</div>
                <h2>Select a steering vector and enter a prompt</h2>
                <p>
                  Compare three approaches side by side: raw baseline, system prompt,
                  and activation vector steering.
                </p>
              </div>
            ) : (
              <>
                <div className="result-panel steered">
                  <div className="result-header">
                    <span className="dot" />
                    <span>Steered</span>
                    <span className="tag">
                      {selected ? `${selected.label} · L${layer} · S${strength}` : ''}
                    </span>
                  </div>
                  <div ref={steeredRef} className={`result-body ${!steeredText && phase !== 'steered' ? 'empty' : ''}`}>
                    {steeredText || (phase === 'steered' ? '' : 'Waiting...')}
                    {phase === 'steered' && <span className="cursor" />}
                  </div>
                </div>
                <div className="result-panel baseline">
                  <div className="result-header">
                    <span className="dot" />
                    <span>Baseline</span>
                    <span className="tag">no steering</span>
                  </div>
                  <div ref={baselineRef} className={`result-body ${!baselineText && phase !== 'baseline' ? 'empty' : ''}`}>
                    {baselineText || (phase === 'baseline' ? '' : (phase === 'steered' ? 'Waiting...' : ''))}
                    {phase === 'baseline' && <span className="cursor" />}
                  </div>
                </div>
                <div className="result-panel prompted">
                  <div className="result-header">
                    <span className="dot" />
                    <span>Prompted</span>
                    <span className="tag">system prompt</span>
                  </div>
                  <div ref={promptedRef} className={`result-body ${!promptedText && phase !== 'prompted' ? 'empty' : ''}`}>
                    {promptedText || (phase === 'prompted' ? '' : (phase !== 'done' ? 'Waiting...' : ''))}
                    {phase === 'prompted' && <span className="cursor" />}
                  </div>
                </div>
              </>
            )}
          </div>
        </main>
      </div>
      )}
    </>
  )
}

function VectorCard({ vec, active, onClick }) {
  return (
    <div className={`vector-card ${active ? 'active' : ''}`} onClick={onClick}>
      <div className="vector-emoji">{vec.emoji}</div>
      <div className="vector-info">
        <div className="vector-label">{vec.label}</div>
        <div className="vector-desc">{vec.description}</div>
      </div>
    </div>
  )
}

export default App
