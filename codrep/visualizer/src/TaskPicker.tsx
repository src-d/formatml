import React from "react";
import Card from "react-bootstrap/Card";
import Form from "react-bootstrap/Form";
import InputGroup from "react-bootstrap/InputGroup";

export interface IProps {
  dataset: string;
  inputTask: string | undefined;
  numberOfTasks: number;
  onInputTaskChange(e: React.ChangeEvent<any>): void;
}

const TaskPicker = (props: IProps) => {
  const max = props.numberOfTasks - 1;
  return (
    <Card>
      <Card.Header>Task picker</Card.Header>
      <Card.Body>
        <Form.Group>
          <InputGroup>
            <InputGroup.Prepend>
              <InputGroup.Text>{props.dataset}/</InputGroup.Text>
            </InputGroup.Prepend>
            <Form.Control
              type="number"
              min={0}
              max={max}
              placeholder="Enter task number"
              value={props.inputTask}
              onChange={props.onInputTaskChange}
            />
            <InputGroup.Append>
              <InputGroup.Text>.txt</InputGroup.Text>
            </InputGroup.Append>
          </InputGroup>
          <Form.Text className="text-muted">
            Enter a task ID between 0 and {max}.
          </Form.Text>
        </Form.Group>
      </Card.Body>
    </Card>
  );
};

export default TaskPicker;
