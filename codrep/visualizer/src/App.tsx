import "./css/App.css";
import Code from "./Code";
import TaskPicker from "./TaskPicker";
import React, { useEffect, useState } from "react";
import Col from "react-bootstrap/Col";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";

export interface IProps {
  endpoint: string;
}

const App = (props: IProps) => {
  const [data, setData] = useState<Data>({
    dataset: "",
    tasks: []
  });
  const [selectedTaskIndex, setSelectedTaskIndex] = useState<number | null>(
    null
  );
  const [task, setTask] = useState<Task | null>(null);
  const [inputTask, setInputTask] = useState<string>("");
  const [message, setMessage] = useState("Loading…");
  const checkInputTask = (task: number) =>
    !Number.isNaN(task) && task >= 0 && task < data.tasks.length;
  const onInputTaskChange = (e: React.ChangeEvent<any>) => {
    setInputTask(e.target.value);
    const newTask = parseInt(e.target.value);
    if (checkInputTask(newTask)) {
      setSelectedTaskIndex(newTask);
    }
  };
  useEffect(() => {
    fetch(props.endpoint + "/tasks", {
      method: "GET",
      mode: "cors"
    })
      .then(response => {
        if (!response.ok) {
          throw Error(response.statusText);
        }
        return response.json();
      })
      .then((data: Data) => {
        if (data.tasks.length === 0) {
          throw Error("Empty response.");
        }
        setData(data);
      })
      .catch(error => {
        setMessage(`Failed to load data with error ${error}.`);
      });
  }, [props.endpoint]);
  useEffect(() => {
    if (selectedTaskIndex === null) {
      return;
    }
    fetch(props.endpoint + "/task", {
      body: JSON.stringify({
        task: selectedTaskIndex
      }),
      method: "POST",
      headers: {
        "Content-Type": "application/json; charset=utf-8"
      },
      mode: "cors"
    })
      .then(response => {
        if (!response.ok) {
          throw Error(response.statusText);
        }
        return response.json();
      })
      .then((task: Task) => {
        setTask(task);
      })
      .catch(error => {
        setMessage(`Failed to load data with error ${error}.`);
      });
  }, [selectedTaskIndex, props.endpoint]);
  if (inputTask !== null) {
    return (
      <Container>
        <Row>
          <Col>
            <TaskPicker
              dataset={data.dataset}
              inputTask={inputTask}
              numberOfTasks={data.tasks.length - 1}
              onInputTaskChange={onInputTaskChange}
            />
          </Col>
        </Row>
        <Row className="code">
          <Col>
            {task !== null ? (
              <Code
                code={task.content}
                error_offset={task.error_offset}
                ranking={task.ranking}
              />
            ) : (
              <div>Please select a task.</div>
            )}
          </Col>
        </Row>
      </Container>
    );
  } else {
    return <div>{message}</div>;
  }
};

export default App;
