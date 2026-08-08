export const SHELL = `<div class="wrap">
  <header>
    <div class="brand">🧠 Neuro<span>Speak</span></div>
    <div class="header-right">
      <span class="chip" id="streakChip">🔥 0-day streak</span>
      <button class="btn" id="settingsBtn">⚙️ API key</button>
    </div>
  </header>

  <div class="hero">
    <h1>Neuroplasticity &amp; Articulation Coach</h1>
    <p>Five minutes a day rewires how you speak. Warm up your voice, think on your feet,
       cut the fluff, and build structures that stick — with instant coaching on every rep.</p>
  </div>

  <nav class="tabs" id="tabs"></nav>
  <main id="main"></main>

  <footer>Your API key and progress never leave this browser · localStorage only</footer>
</div>

<div class="modal-bg" id="modalBg">
  <div class="modal">
    <h3>OpenAI API key</h3>
    <p>Feedback and voice transcription call the OpenAI API <strong>directly from your browser</strong>.
       Your key is stored only in this browser's localStorage and sent only to api.openai.com.
       Get a key at <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener">platform.openai.com/api-keys</a>.</p>
    <input type="password" id="keyInput" placeholder="sk-...">
    <div class="row">
      <button class="btn btn-primary" id="saveKey">Save</button>
      <button class="btn" id="closeModal">Close</button>
      <button class="btn btn-ghost" id="clearKey">Remove key</button>
    </div>
  </div>
</div>
`;
