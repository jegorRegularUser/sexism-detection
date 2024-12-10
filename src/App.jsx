import { useState } from "react";
import "./App.css";

function App() {
  const [load, setLoad] = useState(false);
  const [text, setText] = useState("");
  const [res, setRes] = useState('');
  const [model, setModel] = useState("local_3");

  const handleChange = (event) => {
    setText(event.target.value);
  };

  const handleModelChange = (event) => {
    setModel(event.target.value);
  };
  const checkHandler = async () => {
    setLoad(true);
     await fetch("https://sexism-detection.onrender.com/predict/", { // https://sexism-detection.onrender.com/predict/ http://localhost:8000/predict/
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, model })
    })
      .then(res => res.json())
      .then(res => {
        setRes(res.prediction === 1 ? "Sexist" : "Not Sexist");
      })
      .catch((e) => {
        console.error(e);
      })
      .finally(() => {
        setLoad(false);
      });
  };

  return (
    <>
      <h1>Let's check you on sexism!</h1>
      <p>Please write message below</p>

      <div className="input">
        <select value={model} onChange={handleModelChange}>
          <option value="local_3">Local Model 3e</option>
          <option value="local_5">Local Model 5e</option>
          <option value="collab_5">Collab Model 5e</option>
          <option value="collab_7">Collab Model 7e</option>
          <option value="2_collab_3">2.0 Collab Model 3e</option>
        </select>
        
        <textarea
          value={text}
          onChange={handleChange}
          placeholder="Here..."
        ></textarea>
        <button onClick={checkHandler}>Check!</button>
      </div>

      {load && <p>Loading...</p>}
      {!load && (res ? <p>Your message is {res}</p> : <p>Sorry, try again</p>)}
    </>
  );
}

export default App;
