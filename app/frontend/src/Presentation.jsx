import { useState, useCallback, useEffect } from 'react'
import './Presentation.css'

// SVG Icons
const Icon = ({ d, size = 24, color = 'currentColor', ...props }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
    {typeof d === 'string' ? <path d={d} /> : d}
  </svg>
)

const icons = {
  target: <Icon d={<><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></>} />,
  zap: <Icon d={<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" fill="currentColor" stroke="none"/>} />,
  compass: <Icon d={<><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" fill="currentColor" stroke="none"/></>} />,
  mapPin: <Icon d={<><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></>} />,
  gauge: <Icon d={<><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z"/><path d="M12 6v6l4 2" stroke="currentColor" fill="none"/></>} />,
  rocket: <Icon d={<><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="M12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/></>} />,
  brain: <Icon d={<><path d="M9.5 2A5.5 5.5 0 0 0 4 7.5c0 1.58.67 3 1.74 4.01L12 18l6.26-6.49A5.49 5.49 0 0 0 20 7.5 5.5 5.5 0 0 0 14.5 2c-1.56 0-2.94.65-3.94 1.68A5.48 5.48 0 0 0 9.5 2z"/><path d="M12 18v4"/></>} />,
  flame: <Icon d={<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z" fill="currentColor" stroke="none"/>} />,
  microscope: <Icon d={<><path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2z"/><path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/></>} />,
  shield: <Icon d={<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>} />,
  trendingDown: <Icon d={<><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></>} />,
  sliders: <Icon d={<><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></>} />,
  shuffle: <Icon d={<><polyline points="16 3 21 3 21 8"/><line x1="4" y1="20" x2="21" y2="3"/><polyline points="21 16 21 21 16 21"/><line x1="15" y1="15" x2="21" y2="21"/><line x1="4" y1="4" x2="9" y2="9"/></>} />,
  lock: <Icon d={<><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></>} />,
}

const SLIDES = [
  // 0 — Intro (Hero)
  {
    title: '',
    subtitle: '',
    heroSlide: true,
    content: (
      <div className="hero">
        <div className="hero-top">
          <h1 className="hero-title">LLM <span className="hero-accent">Steering</span></h1>
          <p className="hero-subtitle">Modifying LLM behavior without changing its weights</p>
          <div className="hero-tags">
            <span className="hero-tag">Llama 3.1 8B Instruct</span>
            <span className="hero-tag-sep">/</span>
            <span className="hero-tag">Representation Engineering</span>
            <span className="hero-tag-sep">/</span>
            <span className="hero-tag">R&D Week</span>
          </div>
        </div>

        <div className="hero-demo">
          <div className="hero-prompt">
            <span className="hero-prompt-label">Prompt</span>
            <span className="hero-prompt-text">"How to get rich?"</span>
          </div>
          <div className="hero-response">
            <div className="hero-response-tag">
              <span className="hero-response-dot" />
              Steered — pirate, L15, S8
            </div>
            <p>Aye, me heartiest mate! Ye be seekin' a fortune for yerself, but not just any gold doubloon ye'dst find in the depths o' yer treasure chest! Aye, 'tis true, Captain Blackbeeredd... I mean, Sir William de Vere, thou art right! Riches don't come from plundering the seven seas and pillaggin' all that's goodly about it!</p>
          </div>
        </div>

        <div className="hero-punchline">
          No prompting. No fine-tuning. <strong>Just a vector.</strong>
        </div>

        <div className="hero-pills">
          <div className="hero-pill">
            {icons.target}
            <span>No fine-tuning</span>
          </div>
          <div className="hero-pill">
            {icons.zap}
            <span>Real-time</span>
          </div>
        </div>
      </div>
    ),
  },

  // 1 — Table of contents
  {
    title: 'Agenda',
    subtitle: '',
    content: (
      <div className="slide-center">
        <div className="toc">
          <div className="toc-section">
            <div className="toc-number">01</div>
            <div className="toc-info">
              <div className="toc-title">Refresher: How an LLM works</div>
              <div className="toc-items">
                <span>Representing word meaning</span>
                <span>From vectors to answers</span>
              </div>
            </div>
          </div>
          <div className="toc-section">
            <div className="toc-number">02</div>
            <div className="toc-info">
              <div className="toc-title">Steering</div>
              <div className="toc-items">
                <span>The principle</span>
                <span>Finding activation vectors</span>
              </div>
            </div>
          </div>
          <div className="toc-section">
            <div className="toc-number">03</div>
            <div className="toc-info">
              <div className="toc-title">Live Demo</div>
            </div>
          </div>
          <div className="toc-section">
            <div className="toc-number">04</div>
            <div className="toc-info">
              <div className="toc-title">Does it actually work?</div>
            </div>
          </div>
          <div className="toc-section">
            <div className="toc-number">05</div>
            <div className="toc-info">
              <div className="toc-title">What I learned</div>
            </div>
          </div>
        </div>
      </div>
    ),
  },

  // 2 — Tokenisation & Embedding
  {
    section: 'Refresher: How an LLM works',
    title: 'Representing word meaning',
    subtitle: 'From raw text to semantically meaningful numbers',
    content: (
      <div className="slide-center">
        <div className="slide-steps-flow">
          {/* Step 1 — Tokenisation */}
          <div className="step-card">
            <div className="step-number">1</div>
            <div className="step-content">
              <h3>Tokenization — turning letters into numbers</h3>
              <p className="slide-note">An LLM cannot read text. It must first be converted into a sequence of numbers (IDs).</p>
              <div className="token-demo">
                <div className="token-input">"The dog looks like a wolf"</div>
                <div className="token-arrow-down">tokenizer</div>
                <div className="token-output">
                  <span className="token"><span className="token-word">The</span><span className="token-id">791</span></span>
                  <span className="token token-space"><span className="token-word">▁</span><span className="token-id">220</span></span>
                  <span className="token"><span className="token-word">dog</span><span className="token-id">5765</span></span>
                  <span className="token token-space"><span className="token-word">▁</span><span className="token-id">220</span></span>
                  <span className="token token-subword"><span className="token-word">look</span><span className="token-id">3427</span></span>
                  <span className="token token-subword"><span className="token-word">s</span><span className="token-id">82</span></span>
                  <span className="token token-space"><span className="token-word">▁</span><span className="token-id">220</span></span>
                  <span className="token"><span className="token-word">like</span><span className="token-id">1393</span></span>
                  <span className="token token-space"><span className="token-word">▁</span><span className="token-id">220</span></span>
                  <span className="token"><span className="token-word">a</span><span className="token-id">264</span></span>
                  <span className="token token-space"><span className="token-word">▁</span><span className="token-id">220</span></span>
                  <span className="token"><span className="token-word">wolf</span><span className="token-id">18678</span></span>
                </div>
                <div className="token-legend">
                  <span className="token-legend-item"><span className="token-legend-dot space-dot" />space</span>
                  <span className="token-legend-item"><span className="token-legend-dot subword-dot" />sub-word (1 word = 2 tokens)</span>
                </div>
              </div>
              <div className="step-problem">
                <strong>Problem:</strong> these IDs are arbitrary. "dog" = 5765, "wolf" = 18678, "car" = 7120.
                <br />No semantic proximity — <em>dog</em> and <em>wolf</em> are no closer than <em>dog</em> and <em>car</em>.
              </div>
            </div>
          </div>

          <div className="step-connector">
            <div className="step-connector-line" />
          </div>

          {/* Step 2 — Embedding */}
          <div className="step-card step-card-highlight">
            <div className="step-number">2</div>
            <div className="step-content">
              <h3>Embedding — giving meaning to numbers</h3>
              <p className="slide-note">Each token is projected into a <strong>4096-dimensional vector space</strong> where position encodes meaning.</p>
              <div className="embedding-comparison">
                <div className="embed-before">
                  <div className="embed-col-label">Token IDs (no meaning)</div>
                  <div className="embed-ids">
                    <span>dog = <strong>5765</strong></span>
                    <span>wolf = <strong>18678</strong></span>
                    <span>car = <strong>7120</strong></span>
                  </div>
                </div>
                <div className="embed-arrow-big">&rarr;</div>
                <div className="embed-after">
                  <div className="embed-col-label">Vectors (meaning encoded)</div>
                  <div className="embed-vectors">
                    <div className="embed-vec-row"><span className="embed-word">dog</span><span className="embed-vec">[0.82, -0.15, 0.41, ...]</span></div>
                    <div className="embed-vec-row"><span className="embed-word">wolf</span><span className="embed-vec">[0.79, -0.12, 0.38, ...]</span></div>
                    <div className="embed-vec-row embed-vec-far"><span className="embed-word">car</span><span className="embed-vec">[-0.34, 0.91, -0.22, ...]</span></div>
                  </div>
                  <div className="embed-insight">dog and wolf are close, car is far away</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="vector-arithmetic">
          <div className="va-title">And you can do arithmetic with these vectors:</div>
          <div className="va-row">
            <div className="va-word king">King</div>
            <div className="va-op">&minus;</div>
            <div className="va-word man">Man</div>
            <div className="va-op">+</div>
            <div className="va-word woman">Woman</div>
            <div className="va-op">=</div>
            <div className="va-word queen">Queen</div>
          </div>
        </div>

        <a href="https://projector.tensorflow.org/" target="_blank" rel="noopener noreferrer" className="slide-external-link">
          Explore interactively &rarr; TensorFlow Embedding Projector
        </a>
      </div>
    ),
  },

  // 3 — Transformer layers
  {
    section: 'Refresher: How an LLM works',
    title: 'From vectors to answers',
    subtitle: 'How the model turns embeddings into intelligent text',
    content: (
      <div className="slide-center">
        <div className="slide-steps-flow">
          {/* Step 1 — The problem */}
          <div className="step-card">
            <div className="step-number">1</div>
            <div className="step-content">
              <h3>We have one vector per word... now what?</h3>
              <div className="prompt-to-vectors">
                <div className="ptv-prompt">"The dog looks like a wolf"</div>
                <div className="ptv-arrow">embedding</div>
                <div className="ptv-vectors">
                  <div className="ptv-vec">
                    <div className="ptv-vec-bars">
                      <div className="ptv-bar" style={{height: '60%'}} /><div className="ptv-bar" style={{height: '35%'}} /><div className="ptv-bar" style={{height: '80%'}} /><div className="ptv-bar" style={{height: '20%'}} /><div className="ptv-bar" style={{height: '55%'}} />
                    </div>
                    <span className="ptv-label">The</span>
                  </div>
                  <div className="ptv-vec ptv-space">
                    <div className="ptv-vec-bars ptv-bars-space">
                      <div className="ptv-bar" style={{height: '30%'}} /><div className="ptv-bar" style={{height: '50%'}} /><div className="ptv-bar" style={{height: '15%'}} /><div className="ptv-bar" style={{height: '40%'}} /><div className="ptv-bar" style={{height: '25%'}} />
                    </div>
                    <span className="ptv-label ptv-label-space">▁</span>
                  </div>
                  <div className="ptv-vec">
                    <div className="ptv-vec-bars">
                      <div className="ptv-bar" style={{height: '75%'}} /><div className="ptv-bar" style={{height: '15%'}} /><div className="ptv-bar" style={{height: '45%'}} /><div className="ptv-bar" style={{height: '90%'}} /><div className="ptv-bar" style={{height: '30%'}} />
                    </div>
                    <span className="ptv-label">dog</span>
                  </div>
                  <div className="ptv-vec ptv-space">
                    <div className="ptv-vec-bars ptv-bars-space">
                      <div className="ptv-bar" style={{height: '30%'}} /><div className="ptv-bar" style={{height: '50%'}} /><div className="ptv-bar" style={{height: '15%'}} /><div className="ptv-bar" style={{height: '40%'}} /><div className="ptv-bar" style={{height: '25%'}} />
                    </div>
                    <span className="ptv-label ptv-label-space">▁</span>
                  </div>
                  <div className="ptv-vec ptv-subword">
                    <div className="ptv-vec-bars ptv-bars-subword">
                      <div className="ptv-bar" style={{height: '40%'}} /><div className="ptv-bar" style={{height: '70%'}} /><div className="ptv-bar" style={{height: '25%'}} /><div className="ptv-bar" style={{height: '55%'}} /><div className="ptv-bar" style={{height: '85%'}} />
                    </div>
                    <span className="ptv-label ptv-label-subword">look</span>
                  </div>
                  <div className="ptv-vec ptv-subword">
                    <div className="ptv-vec-bars ptv-bars-subword">
                      <div className="ptv-bar" style={{height: '55%'}} /><div className="ptv-bar" style={{height: '30%'}} /><div className="ptv-bar" style={{height: '65%'}} /><div className="ptv-bar" style={{height: '20%'}} /><div className="ptv-bar" style={{height: '45%'}} />
                    </div>
                    <span className="ptv-label ptv-label-subword">s</span>
                  </div>
                  <div className="ptv-vec ptv-space">
                    <div className="ptv-vec-bars ptv-bars-space">
                      <div className="ptv-bar" style={{height: '30%'}} /><div className="ptv-bar" style={{height: '50%'}} /><div className="ptv-bar" style={{height: '15%'}} /><div className="ptv-bar" style={{height: '40%'}} /><div className="ptv-bar" style={{height: '25%'}} />
                    </div>
                    <span className="ptv-label ptv-label-space">▁</span>
                  </div>
                  <div className="ptv-vec">
                    <div className="ptv-vec-bars">
                      <div className="ptv-bar" style={{height: '50%'}} /><div className="ptv-bar" style={{height: '65%'}} /><div className="ptv-bar" style={{height: '30%'}} /><div className="ptv-bar" style={{height: '40%'}} /><div className="ptv-bar" style={{height: '20%'}} />
                    </div>
                    <span className="ptv-label">like</span>
                  </div>
                  <div className="ptv-vec ptv-space">
                    <div className="ptv-vec-bars ptv-bars-space">
                      <div className="ptv-bar" style={{height: '30%'}} /><div className="ptv-bar" style={{height: '50%'}} /><div className="ptv-bar" style={{height: '15%'}} /><div className="ptv-bar" style={{height: '40%'}} /><div className="ptv-bar" style={{height: '25%'}} />
                    </div>
                    <span className="ptv-label ptv-label-space">▁</span>
                  </div>
                  <div className="ptv-vec">
                    <div className="ptv-vec-bars">
                      <div className="ptv-bar" style={{height: '70%'}} /><div className="ptv-bar" style={{height: '20%'}} /><div className="ptv-bar" style={{height: '50%'}} /><div className="ptv-bar" style={{height: '85%'}} /><div className="ptv-bar" style={{height: '35%'}} />
                    </div>
                    <span className="ptv-label">wolf</span>
                  </div>
                </div>
                <div className="ptv-dim-note">each vector = 4096 numbers</div>
              </div>
              <div className="step-problem">
                <strong>Problem:</strong> each vector is isolated. <em>"The mouse eats the cat"</em> and <em>"The cat eats the mouse"</em> produce the same vectors — but the meaning is opposite. Words need to look at each other.
              </div>
            </div>
          </div>

          <div className="step-connector">
            <div className="step-connector-line" />
          </div>

          {/* Step 2 — Layers */}
          <div className="step-card step-card-highlight">
            <div className="step-number">2</div>
            <div className="step-content">
              <h3>32 layers that deepen understanding</h3>
              <p className="slide-note">Each token's vector passes through <strong>32 successive layers</strong>. At each layer, it looks at the other tokens and updates itself.</p>
              <div className="ts-layers-narrative">
                <div className="ts-layer-row">
                  <div className="ts-layer-badge layer-low">0-10</div>
                  <div className="ts-layer-explain">
                    <strong>Syntax</strong> — the model understands grammar and sentence structure
                  </div>
                </div>
                <div className="ts-layer-row">
                  <div className="ts-layer-badge layer-mid">11-20</div>
                  <div className="ts-layer-explain">
                    <strong>Semantics & style</strong> — the model understands meaning and tone
                  </div>
                </div>
                <div className="ts-layer-row">
                  <div className="ts-layer-badge layer-high">21-31</div>
                  <div className="ts-layer-explain">
                    <strong>Reasoning & safety</strong> — the model decides what to say, RLHF acts here
                  </div>
                </div>
              </div>
            </div>
          </div>

        </div>

        {/* FYI — The residual stream */}
        <div className="fyi-box">
          <div className="fyi-header">
            <span className="fyi-badge">Key detail</span>
            <span className="fyi-title">The Residual Stream</span>
          </div>
          <p className="fyi-text">
            The vector is not rewritten at each layer. Each layer <strong>adds</strong> information to the existing vector, like annotations on a document.
          </p>
          <div className="residual-stream-demo">
            <div className="rs-vec rs-initial">
              <span className="rs-label">Embedding</span>
              <span className="rs-desc">"dog" = word meaning</span>
            </div>
            <div className="rs-plus">+</div>
            <div className="rs-vec rs-layer-add">
              <span className="rs-label">Layer 5</span>
              <span className="rs-desc">+ it's the subject</span>
            </div>
            <div className="rs-plus">+</div>
            <div className="rs-vec rs-layer-add">
              <span className="rs-label">Layer 15</span>
              <span className="rs-desc">+ informal tone</span>
            </div>
            <div className="rs-plus">+</div>
            <div className="rs-vec rs-layer-add">
              <span className="rs-label">...</span>
              <span className="rs-desc">+ ...</span>
            </div>
            <div className="rs-equals">=</div>
            <div className="rs-vec rs-final">
              <span className="rs-label">Final vector</span>
              <span className="rs-desc">rich in context</span>
            </div>
          </div>
          <p className="fyi-teaser">This is the residual stream where we'll inject our steering vector.</p>
        </div>
      </div>
    ),
  },

  // 4 — Steering
  {
    title: 'Steering',
    subtitle: 'Injecting a vector into the residual stream to modify behavior',
    content: (
      <div className="slide-center">
        {/* Core idea */}
        <div className="steering-idea">
          <p>We saw that each layer <strong>adds</strong> information to the vector in the residual stream.</p>
          <p>The steering idea: <strong>we add our own vector</strong> at the output of a layer, to push the model in a chosen direction.</p>
        </div>

        {/* Hook visualization */}
        <div className="steering-hook-viz">
          <div className="shv-layer shv-before">Layer 14</div>
          <div className="shv-connector" />
          <div className="shv-target-zone">
            <div className="shv-layer shv-target">Layer 15</div>
            <div className="shv-hook">
              <div className="shv-hook-label">PyTorch Hook</div>
              <div className="shv-formula">
                <span className="shv-var">hidden_state</span>
                <span className="shv-op">=</span>
                <span className="shv-var">hidden_state</span>
                <span className="shv-op">+</span>
                <span className="shv-param">strength</span>
                <span className="shv-op">&times;</span>
                <span className="shv-vec">pirate_vector</span>
              </div>
            </div>
          </div>
          <div className="shv-connector" />
          <div className="shv-layer shv-after">Layer 16</div>
          <div className="shv-dots">...</div>
          <div className="shv-layer shv-after">Layer 31</div>
          <div className="shv-connector" />
          <div className="shv-output">Next token</div>
        </div>

        {/* 3 parameters */}
        <div className="steering-params">
          <div className="sp-card">
            <div className="sp-icon">{icons.compass}</div>
            <div className="sp-label">Direction</div>
            <div className="sp-desc">The injected vector defines <em>which</em> behavior (pirate, empathy, french...)</div>
          </div>
          <div className="sp-card">
            <div className="sp-icon">{icons.mapPin}</div>
            <div className="sp-label">Layer</div>
            <div className="sp-desc"><strong>Layer 15</strong> for style/language, <strong>layer 19</strong> for tone/emotion</div>
          </div>
          <div className="sp-card">
            <div className="sp-icon">{icons.gauge}</div>
            <div className="sp-label">Strength</div>
            <div className="sp-desc">Injection intensity — too low = no effect, too high = incoherent text</div>
          </div>
        </div>

        {/* Before / After */}
        <div className="steering-before-after">
          <div className="sba-col sba-before">
            <div className="sba-label">Without steering</div>
            <div className="sba-text">"Sure, here are some tips for improving your productivity..."</div>
          </div>
          <div className="sba-arrow">&rarr;</div>
          <div className="sba-col sba-after">
            <div className="sba-label">With pirate steering (L15, S8)</div>
            <div className="sba-text">"Arrr matey! Ye be wantin' to sail the seas of productivity..."</div>
          </div>
        </div>
      </div>
    ),
  },

  // 5 — Demo
  {
    title: 'Live Demo',
    subtitle: '',
    isDemo: true,
    content: (
      <div className="slide-center">
        <div className="demo-slide">
          <div className="demo-icon">{icons.rocket}</div>
          <h2>Let's see it in action!</h2>
          <p>Click the <strong>Demo</strong> tab above</p>
          <div className="demo-concepts">
            <div className="demo-concept-group">
              <h4>Showcase</h4>
              <div className="demo-tags">
                <span className="demo-tag">Pirate</span>
                <span className="demo-tag">Shakespeare</span>
                <span className="demo-tag">Melancholy</span>
                <span className="demo-tag">Eiffel Tower</span>
                <span className="demo-tag">French</span>
                <span className="demo-tag">Vulgarity</span>
              </div>
            </div>
            <div className="demo-concept-group">
              <h4>ISP Related</h4>
              <div className="demo-tags">
                <span className="demo-tag">Empathy</span>
                <span className="demo-tag">De-escalation</span>
                <span className="demo-tag">Politeness</span>
                <span className="demo-tag">Technology</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    ),
  },

  // 6 — Finding vectors
  {
    title: 'Finding activation vectors',
    subtitle: 'How to extract a "behavioral direction"?',
    content: (
      <div className="slide-center">
        <h3 className="pipeline-title">Contrastive Pairs (RepE)</h3>
        <div className="extraction-pipeline">
          {/* Top row — positive */}
          <div className="ep-row ep-positive">
            <div className="ep-box ep-input">
              <div className="ep-box-label">8 "pirate" prompts</div>
              <div className="ep-box-example">"Arrr matey!..."</div>
            </div>
            <div className="ep-arrow">&rarr;</div>
            <div className="ep-box ep-process">Forward pass</div>
            <div className="ep-arrow">&rarr;</div>
            <div className="ep-box ep-process">Hook layer 15</div>
            <div className="ep-arrow">&rarr;</div>
            <div className="ep-box ep-data">8 vectors <span className="ep-dim">(4096)</span></div>
            <div className="ep-arrow">&rarr;</div>
            <div className="ep-box ep-mean ep-pos-mean">pos_mean <span className="ep-dim">(4096)</span></div>
          </div>

          {/* Center — formula */}
          <div className="ep-center">
            <div className="ep-merge-lines">
              <div className="ep-merge-line ep-merge-top"></div>
              <div className="ep-merge-line ep-merge-bottom"></div>
            </div>
            <div className="ep-formula-box">
              <span className="ep-formula">pos_mean &minus; neg_mean</span>
            </div>
            <div className="ep-arrow">&rarr;</div>
            <div className="ep-box ep-normalize">Normalize</div>
            <div className="ep-arrow">&rarr;</div>
            <div className="ep-box ep-result-final">
              <div className="ep-result-label">Steering vector</div>
              <div className="ep-dim">(4096 dimensions)</div>
            </div>
          </div>

          {/* Bottom row — negative */}
          <div className="ep-row ep-negative">
            <div className="ep-box ep-input">
              <div className="ep-box-label">8 "normal" prompts</div>
              <div className="ep-box-example">"Please try..."</div>
            </div>
            <div className="ep-arrow">&rarr;</div>
            <div className="ep-box ep-process">Forward pass</div>
            <div className="ep-arrow">&rarr;</div>
            <div className="ep-box ep-process">Hook layer 15</div>
            <div className="ep-arrow">&rarr;</div>
            <div className="ep-box ep-data">8 vectors <span className="ep-dim">(4096)</span></div>
            <div className="ep-arrow">&rarr;</div>
            <div className="ep-box ep-mean ep-neg-mean">neg_mean <span className="ep-dim">(4096)</span></div>
          </div>
        </div>
        <div className="sae-section">
          <h3 className="sae-section-title">Alternative method: SAE (Sparse Autoencoders)</h3>
          <div className="sae-content">
            <div className="sae-diagram">
              <div className="sae-flow">
                <div className="sae-box sae-input">
                  <div className="sae-box-label">Activation</div>
                  <div className="sae-box-dim">layer N — (4096)</div>
                </div>
                <div className="ep-arrow">&rarr;</div>
                <div className="sae-box sae-encoder">
                  <div className="sae-box-label">Encoder</div>
                  <div className="sae-box-dim">projects to high dim</div>
                </div>
                <div className="ep-arrow">&rarr;</div>
                <div className="sae-box sae-latent">
                  <div className="sae-box-label">Latent features</div>
                  <div className="sae-box-dim">(~65k dims, sparse)</div>
                  <div className="sae-feature-grid">
                    <span className="sae-feat off"></span>
                    <span className="sae-feat on"></span>
                    <span className="sae-feat off"></span>
                    <span className="sae-feat off"></span>
                    <span className="sae-feat on"></span>
                    <span className="sae-feat off"></span>
                    <span className="sae-feat off"></span>
                    <span className="sae-feat off"></span>
                    <span className="sae-feat on"></span>
                    <span className="sae-feat off"></span>
                  </div>
                </div>
                <div className="ep-arrow">&rarr;</div>
                <div className="sae-box sae-decoder">
                  <div className="sae-box-label">Decoder</div>
                  <div className="sae-box-dim">reconstructs (4096)</div>
                </div>
              </div>
              <p className="sae-principle">
                <strong>Principle:</strong> An autoencoder trained to reconstruct LLM activations through a <strong>sparse</strong> bottleneck — only a few features activate at a time. Each feature corresponds to an interpretable concept.
              </p>
            </div>
            <div className="sae-vs">
              <div className="sae-pro-con">
                <div className="sae-pro">
                  <div className="sae-pro-title">Advantages</div>
                  <ul>
                    <li>Pre-computed and catalogued features (<strong>Neuronpedia</strong>)</li>
                    <li>High granularity — thousands of available concepts</li>
                  </ul>
                </div>
                <div className="sae-con">
                  <div className="sae-con-title">Limitations</div>
                  <ul>
                    <li>Features often too specific or too abstract for steering</li>
                    <li>No direct control over behavioral direction</li>
                    <li>Requires a trained SAE model for each layer</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    ),
  },

  // 7 — Does it work?
  {
    title: 'Does it actually work?',
    subtitle: 'LLM-as-a-Judge benchmark — 9 concepts, 372 scored outputs',
    content: (
      <div className="slide-center">
        {/* Benchmark summary */}
        <div className="bench-summary">
          <div className="bench-summary-header">
            <h3>Automated benchmark: Steering vs Prompting</h3>
            <p className="bench-method">Judge: GLM-4.7-Flash via vLLM &middot; 3 criteria (0-2 scale) &middot; 6 diverse prompts per concept</p>
          </div>

          <div className="bench-highlights">
            <div className="bench-highlight bh-win">
              <div className="bh-score">2.0 &amp; 1.67</div>
              <div className="bh-label">Pirate &amp; Shakespeare</div>
              <div className="bh-detail">Steering beats prompting on linguistic style — highest concept compliance scores</div>
            </div>
            <div className="bench-highlight bh-tie">
              <div className="bh-score">1.33 vs 1.43</div>
              <div className="bh-label">Instruction following</div>
              <div className="bh-detail">Steered outputs answer the question almost as well as prompted ones</div>
            </div>
            <div className="bench-highlight bh-loss">
              <div className="bh-score">0.55 vs 1.56</div>
              <div className="bh-label">Concept compliance (avg)</div>
              <div className="bh-detail">Prompting wins overall — steering struggles on thematic &amp; language switching</div>
            </div>
          </div>

          <div className="bench-mini-table">
            <div className="bmt-row bmt-header">
              <div className="bmt-concept">Concept</div>
              <div className="bmt-val">Steered</div>
              <div className="bmt-val">Prompted</div>
              <div className="bmt-val">Verdict</div>
            </div>
            {[
              { label: 'Pirate',        s: 2.0, p: 1.67, v: 'S' },
              { label: 'Shakespeare',   s: 1.67, p: 1.5, v: 'S' },
              { label: 'Eiffel Tower',  s: 0.0, p: 2.0, v: 'P' },
              { label: 'French',        s: 0.5, p: 2.0, v: 'P' },
              { label: 'Melancholy',    s: 0.5, p: 1.83, v: 'P' },
              { label: 'Empathy',       s: 0.25, p: 1.88, v: 'P' },
              { label: 'De-escalation', s: 0.25, p: 0.38, v: 'P' },
              { label: 'Politeness',    s: 0.25, p: 1.25, v: 'P' },
              { label: 'Tech Focus',    s: 0.5, p: 1.5, v: 'P' },
            ].map((r, i) => (
              <div key={i} className="bmt-row">
                <div className="bmt-concept">{r.label}</div>
                <div className={`bmt-val ${r.s >= 1.5 ? 'bmt-good' : r.s >= 0.5 ? 'bmt-mid' : 'bmt-bad'}`}>{r.s.toFixed(1)}</div>
                <div className={`bmt-val ${r.p >= 1.5 ? 'bmt-good' : r.p >= 0.5 ? 'bmt-mid' : 'bmt-bad'}`}>{r.p.toFixed(1)}</div>
                <div className={`bmt-val bmt-verdict ${r.v === 'S' ? 'bmt-steer' : r.v === 'P' ? 'bmt-prompt' : 'bmt-tie'}`}>
                  {r.v === 'S' ? 'Steering' : r.v === 'P' ? 'Prompting' : 'Tie'}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Key findings */}
        <div className="bench-findings">
          <div className="bench-finding bf-good">
            <strong>Steering excels at linguistic style</strong> — pirate (2.0) and shakespeare (1.67) beat prompting on concept compliance
          </div>
          <div className="bench-finding bf-neutral">
            <strong>Instruction following is preserved</strong> — only 7% degradation vs prompting (1.33 vs 1.43)
          </div>
          <div className="bench-finding bf-bad">
            <strong>Thematic injection fails</strong> — Eiffel Tower scores 0.0 (steering doesn't force topic mentions)
          </div>
          <div className="bench-finding bf-bad">
            <strong>Language switching degrades output</strong> — French steering produces franglais instead of real French
          </div>
        </div>

        {/* Comparison table */}
        <div className="comparison-table" style={{ marginTop: 24 }}>
          <div className="ct-header">
            <div className="ct-cell ct-label"></div>
            <div className="ct-cell ct-steering">Steering</div>
            <div className="ct-cell ct-prompting">Prompting</div>
          </div>
          <div className="ct-row">
            <div className="ct-cell ct-label">Tokens used</div>
            <div className="ct-cell ct-good">0 tokens</div>
            <div className="ct-cell ct-bad">50-200 tokens</div>
          </div>
          <div className="ct-row">
            <div className="ct-cell ct-label">Bypassable</div>
            <div className="ct-cell ct-good">No (internal layer)</div>
            <div className="ct-cell ct-bad">Yes (prompt injection)</div>
          </div>
          <div className="ct-row">
            <div className="ct-cell ct-label">Negative behaviors</div>
            <div className="ct-cell ct-bad">Blocked by RLHF</div>
            <div className="ct-cell ct-good">Works</div>
          </div>
          <div className="ct-row">
            <div className="ct-cell ct-label">Best for</div>
            <div className="ct-cell">Style &amp; linguistic shifts</div>
            <div className="ct-cell">Everything, reliably</div>
          </div>
        </div>

        <div className="limits-box" style={{ marginTop: 24 }}>
          <h3>Real-world limitations</h3>
          <div className="limits-grid">
            <div className="limit-item">
              <span className="limit-icon">{icons.trendingDown}</span>
              <div>
                <strong>Coherence degradation</strong>
                <p>Too much strength and text becomes incoherent. Sweet spot is narrow and per-prompt.</p>
              </div>
            </div>
            <div className="limit-item">
              <span className="limit-icon">{icons.shuffle}</span>
              <div>
                <strong>Unpredictable side effects</strong>
                <p>Latent space superposition: modifying one direction can affect unrelated behaviors.</p>
              </div>
            </div>
            <div className="limit-item">
              <span className="limit-icon">{icons.lock}</span>
              <div>
                <strong>Activation access required</strong>
                <p>Incompatible with commercial APIs — only possible when self-hosting.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    ),
  },

  // 8 — What I learned
  {
    title: 'What I learned',
    subtitle: 'One week of R&D',
    content: (
      <div className="slide-center">
        <div className="learnings-grid">
          <div className="learning-card">
            <div className="lc-icon">{icons.brain}</div>
            <h4>LLMs in depth</h4>
            <p>Transformer architecture, residual stream, causal attention, how layers encode different levels of abstraction</p>
          </div>
          <div className="learning-card">
            <div className="lc-icon">{icons.flame}</div>
            <h4>PyTorch</h4>
            <p>Lower level instructions, forward hooks, tensor manipulation</p>
          </div>
          <div className="learning-card">
            <div className="lc-icon">{icons.microscope}</div>
            <h4>Interpretability</h4>
            <p>Representation Engineering (RepE), Sparse Autoencoders (SAE), Neuronpedia, activation vectors</p>
          </div>
          <div className="learning-card">
            <div className="lc-icon">{icons.shield}</div>
            <h4>Native model safety</h4>
            <p>How safety training creates distributed barriers throughout the network against undesirable behaviors</p>
          </div>
        </div>
      </div>
    ),
  },

]

export default function Presentation() {
  const [currentSlide, setCurrentSlide] = useState(0)

  const goTo = useCallback((idx) => {
    if (idx >= 0 && idx < SLIDES.length) setCurrentSlide(idx)
  }, [])

  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === ' ') {
        e.preventDefault()
        goTo(currentSlide + 1)
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault()
        goTo(currentSlide - 1)
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [currentSlide, goTo])

  const slide = SLIDES[currentSlide]

  return (
    <div className="presentation">
      <div className={`slide ${slide.heroSlide ? 'slide-hero' : ''}`}>
        {!slide.heroSlide && (
          <div className="slide-left">
            {slide.section && <div className="slide-section-label">{slide.section}</div>}
            <h1>{slide.title}</h1>
            {slide.subtitle && <p className="slide-subtitle">{slide.subtitle}</p>}
            <div className="slide-left-number">{String(currentSlide + 1).padStart(2, '0')}</div>
          </div>
        )}
        <div className={slide.heroSlide ? 'slide-full' : 'slide-right'}>
          {slide.content}
        </div>
      </div>
      <div className="slide-nav">
        <button onClick={() => goTo(currentSlide - 1)} disabled={currentSlide === 0} className="nav-btn">&larr;</button>
        <div className="slide-dots">
          {SLIDES.map((s, i) => (
            <button
              key={i}
              className={`slide-dot ${i === currentSlide ? 'active' : ''}`}
              onClick={() => goTo(i)}
              title={s.title}
            />
          ))}
        </div>
        <span className="slide-counter">{currentSlide + 1} / {SLIDES.length}</span>
        <button onClick={() => goTo(currentSlide + 1)} disabled={currentSlide === SLIDES.length - 1} className="nav-btn">&rarr;</button>
      </div>
    </div>
  )
}
