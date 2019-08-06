import "./css/App.css";
import Code from "./Code";
import Metadata from "./Metadata";
import React, { useEffect, useReducer } from "react";
import Col from "react-bootstrap/Col";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";

export interface IAppProps {
  endpoint: string;
}

export interface IAppState {
  numberOfTasks: number;
  metadata: Metadata;
  selectedTaskIndex: number | null;
  selectedTask: Task | null;
  selectedOffset: number | null;
  inputTask: string;
  message: string;
}

export enum ActionTarget {
  numberOfTasks = "numberOfTasks",
  metadata = "metadata",
  selectedTaskIndex = "selectedTaskIndex",
  selectedTask = "selectedTask",
  selectedOffset = "selectedOffset",
  inputTask = "inputTask",
  message = "message"
}

const initialState = {
  numberOfTasks: 0,
  metadata: {
    columns: [],
    metadata: {}
  },
  selectedTaskIndex: null,
  selectedTask: null,
  selectedOffset: null,
  inputTask: "",
  message: "Loading…",
  task: null
};

const reducer = (
  state: IAppState,
  action: { target: ActionTarget; payload: any }
): IAppState => {
  switch (action.target) {
    case ActionTarget.inputTask:
      const newState = { ...state, [action.target]: action.payload };
      const newTask = parseInt(action.payload);
      if (
        !Number.isNaN(newTask) &&
        newTask >= 0 &&
        newTask < state.numberOfTasks
      ) {
        newState[ActionTarget.selectedTaskIndex] = newTask;
      }
      return newState;
    default:
      return { ...state, [action.target]: action.payload };
  }
};

export const AppContext = React.createContext<
  (React.Dispatch<{ target: ActionTarget; payload: any }>) | null
>(null);

const App = (props: IAppProps) => {
  const [state, dispatch] = useReducer(reducer, initialState);
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
        if (data.numberOfTasks === 0) {
          throw Error("Empty response.");
        }
        dispatch({
          target: ActionTarget.numberOfTasks,
          payload: data.numberOfTasks
        });
        dispatch({
          target: ActionTarget.metadata,
          payload: data.metadata
        });
        dispatch({
          target: ActionTarget.inputTask,
          payload: "0"
        });
        dispatch({
          target: ActionTarget.selectedTaskIndex,
          payload: 0
        });
      })
      .catch(error => {
        dispatch({
          target: ActionTarget.message,
          payload: `Failed to load data with error ${error}.`
        });
      });
  }, [props.endpoint]);
  useEffect(() => {
    if (state.selectedTaskIndex === null) {
      return;
    }
    fetch(props.endpoint + "/task", {
      body: JSON.stringify({
        task: state.selectedTaskIndex
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
        dispatch({ target: ActionTarget.selectedTask, payload: task });
        dispatch({
          target: ActionTarget.selectedOffset,
          payload: task.errorOffset
        });
      })
      .catch(error => {
        dispatch({
          target: ActionTarget.message,
          payload: `Failed to load data with error ${error}.`
        });
      });
  }, [state.selectedTaskIndex, props.endpoint]);
  if (state.inputTask !== null) {
    let metadata: string[] = [];
    if (
      state.selectedOffset !== null &&
      state.metadata.metadata.hasOwnProperty(
        `${state.selectedTaskIndex}.txt`
      ) &&
      state.metadata.metadata[`${state.selectedTaskIndex}.txt`].hasOwnProperty(
        state.selectedOffset
      )
    ) {
      metadata =
        state.metadata.metadata[`${state.selectedTaskIndex}.txt`][
          state.selectedOffset
        ];
    }
    const offsetRank =
      state.selectedTask !== null && state.selectedOffset !== null
        ? state.selectedTask.ranking.indexOf(state.selectedOffset) + 1
        : null;
    return (
      <AppContext.Provider value={dispatch}>
        <Container fluid={true}>
          <Row>
            <Col sm={9}>
              {state.selectedTask !== null ? (
                <Code
                  inputTask={state.inputTask}
                  numberOfTasks={state.numberOfTasks}
                  code={state.selectedTask.content}
                  errorOffset={state.selectedTask.errorOffset}
                  selectedOffset={state.selectedOffset}
                  ranking={state.selectedTask.ranking}
                />
              ) : (
                <div>Please select a task.</div>
              )}
            </Col>
            <Col sm={3}>
              <Metadata
                columns={state.metadata.columns}
                values={metadata}
                rank={offsetRank}
              />
            </Col>
          </Row>
        </Container>
      </AppContext.Provider>
    );
  } else {
    return <div>{state.message}</div>;
  }
};

export default App;
