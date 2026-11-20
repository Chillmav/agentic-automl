const {useEffect, useState} = React;

function ResultsTable({rows}) {
  if (!rows.length) return <p>Not evaluated: this follow-up uses CV only.</p>;
  const columns = Object.keys(rows[0]);
  return (
    <table>
      <thead><tr>{columns.map(name => <th key={name} scope="col">{name}</th>)}</tr></thead>
      <tbody>
        {rows.map(row => (
          <tr key={row.model ?? row.column}>
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

function DatasetReview({review, running, onDecision, onCancel}) {
  return (
    <section aria-label="Dataset review">
      <h2>Review dataset before training</h2>
      <p>{review.rows.toLocaleString()} uploaded rows. Training will use at most 10,000 rows.</p>
      <table>
        <thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Distinct</th></tr></thead>
        <tbody>{review.columns.map(column => (
          <tr key={column.name}><td>{column.name}</td><td>{column.dtype}</td>
            <td>{column.missing}</td><td>{column.distinct}</td></tr>
        ))}</tbody>
      </table>
      <h3>Proposed changes</h3>
      {review.proposed_changes.length ? (
        <ul>{review.proposed_changes.map((change, index) => <li key={index}>{change.description}</li>)}</ul>
      ) : <p>No additional changes proposed.</p>}
      <h3>Things to check</h3>
      <ul>{review.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul>
      <p>After this decision, the existing pipeline will still impute missing features, scale numbers,
        encode categories and reject categorical columns with more than 20 distinct training values.
        These steps are fitted separately inside each CV training fold.</p>
      <p>Your source file stays unchanged. Declining these suggestions may leave issues that prevent training.</p>
      <button disabled={running} onClick={() => onDecision(true)}>Accept all and train</button>{' '}
      <button disabled={running} onClick={() => onDecision(false)}>Use original data and train</button>{' '}
      <button disabled={running} onClick={onCancel}>Cancel review</button>
    </section>
  );
}

function App() {
  const [models, setModels] = useState(null);
  const [task, setTask] = useState('classification');
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState('Loading models…');
  const [running, setRunning] = useState(false);
  const [review, setReview] = useState(null);
  const [pendingData, setPendingData] = useState(null);

  useEffect(() => {
    fetch('/api/models').then(response => {
      if (!response.ok) throw new Error('Could not load models. Reload the page to retry.');
      return response.json();
    }).then(data => {
      setModels(data);
      setStatus('');
    }).catch(error => setStatus(error.message));
  }, []);

  async function reviewDataset(event) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    setResult(null);
    if (!data.has('model_names')) {
      setStatus('Choose at least one model.');
      return;
    }
    setRunning(true);
    setStatus('Inspecting dataset…');
    try {
      const response = await fetch('/api/dataset-review', {method: 'POST', body: data});
      const responseData = await response.json();
      if (!response.ok) throw new Error(typeof responseData.detail === 'string'
        ? responseData.detail : JSON.stringify(responseData.detail));
      setReview(responseData);
      setPendingData(data);
      setStatus('Review the suggestions and choose how to continue.');
    } catch (error) {
      setStatus(`Review failed: ${error.message}`);
    } finally {
      setRunning(false);
    }
  }

  async function runExperiment(accepted) {
    const data = pendingData;
    data.set('apply_suggestions', String(accepted));
    setReview(null);
    setPendingData(null);
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
      <form onSubmit={reviewDataset}>
        <fieldset className="form-fields" disabled={running || !!review}>
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
          <button disabled={!models || running}>Review dataset</button>
        </fieldset>
      </form>
      <p id="status" role="status" aria-live="polite">{status}</p>
      {review && <DatasetReview review={review} running={running} onDecision={runExperiment}
        onCancel={() => { setReview(null); setPendingData(null); setStatus('Review cancelled. You can edit the form.'); }} />}
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
      <p>Newest runs first. Follow-ups show CV scores; initial runs also have final test scores.</p>
      {error && <p role="alert">{error}</p>}
      {!experiments && !error && <p role="status">Loading history…</p>}
      {experiments?.length === 0 && <p>No experiments yet. Start a new experiment to save your first result.</p>}
      {experiments?.length > 0 && (
        <div className="history-table">
          <table>
            <thead><tr>
              <th scope="col">Date</th><th scope="col">Dataset</th><th scope="col">Task</th>
              <th scope="col">Best model</th><th scope="col">Scores</th>
            </tr></thead>
            <tbody>
              {experiments.map(experiment => (
                <tr key={experiment.id}>
                  <td><time dateTime={experiment.completed_at}>{new Date(experiment.completed_at).toLocaleString()}</time></td>
                  <td><a href={`/experiments/${encodeURIComponent(experiment.id)}`}>{experiment.dataset}</a></td>
                  <td>{experiment.task}</td>
                  <td>{experiment.selected_model}</td>
                  <td>{experiment.test_results.length ? 'Test: ' : 'CV: '}{Object.entries(experiment.test_results[0]
                    || experiment.cv_results.find(row => row.model === experiment.selected_model))
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
      <AnalystPanel experiment={experiment} />
      {experiment.parent_id && <section>
        <h2>Follow-up comparison</h2>
        <p>CV-only follow-up to <a href={`/experiments/${experiment.parent_id}`}>parent experiment</a>.
          Excluded: {experiment.approved_action.column}.</p>
        <p>{experiment.cv_comparison.metric.toUpperCase()}: parent {experiment.cv_comparison.parent.toFixed(4)},
          follow-up {experiment.cv_comparison.follow_up.toFixed(4)}.
          {' '}{experiment.cv_comparison.improved ? 'Improved on CV.' : 'No improvement on CV.'}</p>
      </section>}
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
      <h2>Training data profile</h2>
      {!experiment.training_profile ? <p>Not recorded for this experiment.</p> : <>
        <p>Computed from {experiment.training_profile.rows} outer training rows. Test rows are excluded.</p>
        {experiment.training_profile.target.class_counts ? <p>Class balance: {Object.entries(experiment.training_profile.target.class_counts)
          .map(([label, count]) => `${label}: ${count} (${(100 * count / experiment.training_profile.rows).toFixed(1)}%)`).join(', ')}</p>
          : <p>Target: {Object.entries(experiment.training_profile.target).map(([name, value]) => `${name}: ${value.toFixed(4)}`).join(', ')}</p>}
        <div className="history-table"><ResultsTable rows={experiment.training_profile.columns.map(column => ({
          column: column.name, type: column.dtype, missing_percent: column.missing_percent,
          distinct: column.distinct, min: column.numeric?.min ?? null, max: column.numeric?.max ?? null,
          mean: column.numeric?.mean ?? null, median: column.numeric?.median ?? null,
        }))} /></div>
      </>}
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
      <h2>Dataset preparation</h2>
      {!experiment.dataset_review ? <p>Not recorded for this experiment.</p> : (
        <>
          <p>{experiment.dataset_review.accepted ? 'Proposed changes accepted.' : 'Original data used; proposed changes declined.'}</p>
          <p>Uploaded rows: {experiment.dataset_review.rows}. After preparation: {experiment.dataset_review.prepared_rows}.</p>
          {experiment.dataset_review.applied_changes.length ? (
            <ul>{experiment.dataset_review.applied_changes.map((change, index) => <li key={index}>{change.description}</li>)}</ul>
          ) : <p>No additional operations applied.</p>}
          <h3>Review notes</h3>
          <ul>{experiment.dataset_review.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul>
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
