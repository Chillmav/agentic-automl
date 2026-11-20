// Report history and one explicitly approved operation. No automatic model calls.
function AnalystPanel({experiment}) {
  const [reports, setReports] = React.useState([]);
  const [selectedId, setSelectedId] = React.useState('');
  const [error, setError] = React.useState('');
  const [busy, setBusy] = React.useState(false);
  const [loading, setLoading] = React.useState(true);
  const [followUp, setFollowUp] = React.useState(null);
  const analysis = reports.find(report => report.id === selectedId);
  const base = `/api/experiments/${encodeURIComponent(experiment.id)}`;

  React.useEffect(() => {
    fetch(`${base}/analyses`).then(async response => {
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Could not load reports.');
      setReports(data);
      setSelectedId(data[0]?.id || '');
    }).catch(error => setError(error.message)).finally(() => setLoading(false));
  }, [experiment.id]);

  async function request(path, body) {
    setBusy(true);
    setError('');
    try {
      const response = await fetch(path, {method: 'POST', body});
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Request failed.');
      return data;
    } catch (error) {
      setError(error.message);
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function analyze() {
    const report = await request(`${base}/analysis`);
    if (report) {
      setReports(previous => [report, ...previous]);
      setSelectedId(report.id);
      setFollowUp(null);
    }
  }

  async function approve(event) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    data.set('analysis_id', analysis.id);
    data.set('approved', 'true');
    const result = await request(`${base}/follow-up`, data);
    if (result) setFollowUp(result);
  }

  async function reject() {
    if (await request(`${base}/analyses/${encodeURIComponent(analysis.id)}/reject`)) {
      setReports(previous => previous.map(report => report.id === analysis.id
        ? {...report, rejected: true} : report));
    }
  }

  return (
    <section aria-label="Experiment analyst">
      <h2>Experiment analyst</h2>
      <p>The analyst uses training profiles and CV scores. Every report is saved; final test scores are excluded.</p>
      <button disabled={busy || loading} onClick={analyze}>Analyze experiment</button>
      {loading && <p role="status">Loading saved reports…</p>}
      {busy && <p role="status">Working…</p>}
      {error && <p role="alert">{error}</p>}
      {reports.length > 0 && <label>Saved report <select value={selectedId} disabled={busy}
        onChange={event => {setSelectedId(event.target.value); setFollowUp(null); setError('');}}>
        {reports.map(report => <option key={report.id} value={report.id}>
          {new Date(report.created_at).toLocaleString()} — {report.model}
        </option>)}
      </select></label>}
      {analysis && <>
        <p>{analysis.summary}</p>
        <h3>Findings</h3>
        <ul>{analysis.findings.map((finding, index) => <li key={index}>{finding}</li>)}</ul>
        <h3>Suggested next steps</h3>
        <ol>{analysis.next_steps.map((step, index) => <li key={index}>{step}</li>)}</ol>
        {analysis.proposed_action && <>
          <h3>Proposed column exclusion</h3>
          <p>Exclude <strong>{analysis.proposed_action.column}</strong>: {analysis.proposed_action.reason}</p>
          {analysis.rejected ? <p>Suggestion rejected. No follow-up will be started from this report.</p>
            : !experiment.dataset_sha256 ? <p>This older experiment has no dataset fingerprint. Create a new experiment to enable follow-ups.</p>
            : <form onSubmit={approve}>
              <fieldset disabled={busy}>
                <legend>Your decision</legend>
                <p>Re-upload the exact original CSV. This run keeps the same models, preparation and seeded splits,
                  excludes this column, and compares CV scores. The test set will not be evaluated again.</p>
                <label>Original CSV <input type="file" name="file" accept=".csv,.xls" required /></label>
                <button>Approve exclusion and run CV</button>{' '}
                <button type="button" onClick={reject}>Reject suggestion</button>
              </fieldset>
            </form>}
        </>}
      </>}
      {followUp && <p role="status">Follow-up saved. <a href={`/experiments/${followUp.id}`}>View CV comparison</a></p>}
    </section>
  );
}
