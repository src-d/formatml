import React, { useContext } from "react";
import Form from "react-bootstrap/Form";
import InputGroup from "react-bootstrap/InputGroup";
import { AppContext, ActionTarget } from "./App";

export interface ITaskPickerProps {
  inputTask: string | undefined;
  numberOfTasks: number;
}

const TaskPicker = (props: ITaskPickerProps) => {
  const max = props.numberOfTasks - 1;
  const dispatch = useContext(AppContext);
  return (
    <Form.Group>
      <InputGroup>
        <Form.Control
          type="number"
          min={0}
          max={max}
          placeholder="Enter task number"
          value={props.inputTask}
          onChange={(e: React.FormEvent<any>) => {
            if (dispatch !== null) {
              return dispatch({
                target: ActionTarget.inputTask,
                payload: e.currentTarget.value
              });
            }
          }}
        />
        <InputGroup.Append>
          <InputGroup.Text>.txt</InputGroup.Text>
        </InputGroup.Append>
      </InputGroup>
      <Form.Text className="text-muted">
        Enter a task ID between 0 and {max}.
      </Form.Text>
    </Form.Group>
  );
};

export default TaskPicker;
