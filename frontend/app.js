const {useEffect, useState} = React;

function ResultsTable({rows}) {
  const columns = Object.keys(rows[0]);
  return (
    <table>
      <thead><tr>{columns.map(name => <th key={name} scope="col">{name}</th>)}</tr></thead>
      <tbody>
        {rows.map(row => (
          <tr key={row.model}>
            {columns.map(name => (
              <td key={name}>
                {typeof row[name] === 'number' ? row[name].toFixed(4) : (row[name] ?? 'Unavailable')}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function App() {
  const [models, setModels] = useState(null);
  const [task, setTask] = useState('classification');
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState('Loading models…');
  const [running, setRunning] = useState(false);

  useEffect(() => {
    fetch('/api/models').then(response => {
      if (!response.ok) throw new Error('Could not load models. Reload the page to retry.');
      return response.json();
    }).then(data => {
      setModels(data);
      setStatus('');
    }).catch(error => setStatus(error.message));
  }, []);

  async function runExperiment(event) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    setResult(null);
    if (!data.has('model_names')) {
      setStatus('Choose at least one model.');
      return;
    }
    setRunning(true);
    setStatus('Running CV and evaluating the winner. This may take a while…');
    try {
      const response = await fetch('/api/experiments', {method: 'POST', body: data});
      const responseData = await response.json();
      if (!response.ok) {
        throw new Error(typeof responseData.detail === 'string'
          ? responseData.detail : JSON.stringify(responseData.detail));
      }
      setResult(responseData);
      setStatus(`Finished. Used ${responseData.sample_rows} rows.`);
    } catch (error) {
      setStatus(`Experiment failed: ${error.message}`);
    } finally {
      setRunning(false);
    }
  }

  return (
    <>
      <h1>AutoML experiments</h1>
      <p>Compare models with 3-fold CV, then test only the winner. Up to 10,000 sampled rows per run.</p>
      <p>This version uses a random 80/20 split. Use independent observations, not time-series data.
        Remove ID columns and features that reveal the target before uploading.</p>
      <form onSubmit={runExperiment}>
        <fieldset className="form-fields" disabled={running}>
          <label>CSV file (up to 25 MB) <input name="file" type="file" accept=".csv,.xls" required /></label>
          <label>Task <select name="task" value={task} onChange={event => {
            setTask(event.target.value);
            setResult(null);
            setStatus('');
          }}>
            <option value="classification">Binary classification</option>
            <option value="regression">Regression</option>
          </select></label>
          <label>Target column <input name="target_column" required placeholder="Exact CSV column name" /></label>
          {task === 'classification' && (
            <label>Positive class <input name="positive_class" defaultValue="1" required /></label>
          )}
          <fieldset key={task}>
            <legend>Models (baselines are always included)</legend>
            {models?.[task].map((name, index) => (
              <label key={name}>
                <input type="checkbox" name="model_names" value={name} defaultChecked={index === 0} /> {name}
              </label>
            ))}
          </fieldset>
          <button disabled={!models || running}>Run experiment</button>
        </fieldset>
      </form>
      <p id="status" role="status" aria-live="polite">{status}</p>
      {result && (
        <section>
          <p><a href={`/experiments/${encodeURIComponent(result.id)}`}>View experiment details</a></p>
          <h2>CV averages</h2>
          <ResultsTable rows={result.cv_results} />
          <h2>Winner on test data</h2>
          <p>{result.selected_model}, selected using {result.selection_metric}.</p>
          <ResultsTable rows={result.test_results} />
        </section>
      )}
    </>
  );
}

function History() {
  const [experiments, setExperiments] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch('/api/experiments').then(response => {
      if (!response.ok) throw new Error('Could not load experiment history. Reload to retry.');
      return response.json();
    }).then(setExperiments).catch(error => setError(error.message));
  }, []);

  return (
    <>
      <h1>Experiment history</h1>
      <p>Newest runs first. Winners are selected by CV; scores below are from the final test set.</p>
      {error && <p role="alert">{error}</p>}
      {!experiments && !error && <p role="status">Loading history…</p>}
      {experiments?.length === 0 && <p>No experiments yet. Start a new experiment to save your first result.</p>}
      {experiments?.length > 0 && (
        <div className="history-table">
          <table>
            <thead><tr>
              <th scope="col">Date</th><th scope="col">Dataset</th><th scope="col">Task</th>
              <th scope="col">Best model</th><th scope="col">Test scores</th>
            </tr></thead>
            <tbody>
              {experiments.map(experiment => (
                <tr key={experiment.id}>
                  <td><time dateTime={experiment.completed_at}>{new Date(experiment.completed_at).toLocaleString()}</time></td>
                  <td><a href={`/experiments/${encodeURIComponent(experiment.id)}`}>{experiment.dataset}</a></td>
                  <td>{experiment.task}</td>
                  <td>{experiment.selected_model}</td>
                  <td>{Object.entries(experiment.test_results[0])
                    .filter(([name]) => name !== 'model')
                    .map(([name, value]) => `${name.toUpperCase()}: ${value == null ? 'Unavailable' : value.toFixed(4)}`)
                    .join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

function ExperimentDetails({experimentId}) {
  const [experiment, setExperiment] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`/api/experiments/${encodeURIComponent(experimentId)}`).then(async response => {
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Could not load experiment.');
      return data;
    }).then(setExperiment).catch(error => setError(error.message));
  }, [experimentId]);

  if (error) return <p role="alert">{error}</p>;
  if (!experiment) return <p role="status">Loading experiment…</p>;

  const rejected = experiment.rejected_columns;
  return (
    <>
      <h1>Experiment details</h1>
      <dl>
        <dt>Dataset</dt><dd>{experiment.dataset}</dd>
        <dt>Completed</dt><dd>{new Date(experiment.completed_at).toLocaleString()}</dd>
        <dt>Task</dt><dd>{experiment.task}</dd>
        <dt>Target column</dt><dd>{experiment.target_column}</dd>
        {experiment.task === 'classification' && (
          <><dt>Positive class</dt><dd>{experiment.positive_class}</dd></>
        )}
        <dt>Sampled rows</dt><dd>{experiment.sample_rows.toLocaleString()}</dd>
        <dt>Selected models</dt><dd>{experiment.model_names.join(', ')}</dd>
        <dt>Winning model</dt><dd>{experiment.selected_model}</dd>
        <dt>Selection metric</dt><dd>{experiment.selection_metric}</dd>
      </dl>
      <h2>CV averages</h2>
      <ResultsTable rows={experiment.cv_results} />
      <h2>Winner on test data</h2>
      <ResultsTable rows={experiment.test_results} />
      <h2>Rejected columns</h2>
      {rejected == null ? (
        <p>Not recorded for this experiment.</p>
      ) : experiment.selected_model.endsWith('_baseline') ? (
        <p>The baseline winner does not use feature preprocessing.</p>
      ) : Object.keys(rejected).length === 0 ? (
        <p>No columns were rejected by the winning model's fitted preprocessor.</p>
      ) : (
        <>
          <p>The winning model's final training split had more than 20 distinct values in these categorical columns.
            Counts may differ between CV folds.</p>
          <ul>{Object.entries(rejected).map(([name, count]) => (
            <li key={name}>{name}: {count} distinct values</li>
          ))}</ul>
        </>
      )}
    </>
  );
}

const experimentId = window.location.pathname.startsWith('/experiments/')
  ? decodeURIComponent(window.location.pathname.split('/')[2]) : null;

ReactDOM.createRoot(document.getElementById('root')).render(
  <>
    <nav aria-label="Main navigation"><a href="/history">History</a> | <a href="/new">New experiment</a></nav>
    {experimentId ? <ExperimentDetails experimentId={experimentId} />
      : window.location.pathname === '/new' ? <App /> : <History />}
  </>
);
