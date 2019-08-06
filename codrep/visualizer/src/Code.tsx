import React from "react";
import Span, { ISpanHandles, SpanType } from "./Span";
import TaskPicker from "./TaskPicker";
import Button from "react-bootstrap/Button";
import ButtonGroup from "react-bootstrap/ButtonGroup";
import ButtonToolbar from "react-bootstrap/ButtonToolbar";
import Card from "react-bootstrap/Card";
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";
import "./css/Code.css";

export interface ICodeProps {
  code: string;
  ranking: number[];
  errorOffset: number;
  inputTask: string | undefined;
  numberOfTasks: number;
  selectedOffset: number | null;
}

const useComputeSpans = (
  code: string,
  ranking: number[],
  errorOffset: number,
  selectedOffset: number | null,
  nPredRefs = 5
): [
  JSX.Element[],
  React.RefObject<ISpanHandles>,
  React.RefObject<ISpanHandles>[]
] => {
  const spans: JSX.Element[] = [];
  const predictions = new Set(ranking);
  const indices = Array.from(new Set([...ranking, errorOffset]));
  const firstPredictions = ranking.slice(0, nPredRefs);
  const toReference = new Set([errorOffset, ...firstPredictions]);
  let errorRef = null;
  let predictionRefs = [];
  indices.sort((a, b) => a - b);
  let currentIndex = 0;
  for (const index of indices) {
    if (currentIndex < index) {
      spans.push(
        <Span
          key={currentIndex}
          code={code.slice(currentIndex, index)}
          types={[SpanType.NotPredicted]}
          selected={false}
        />
      );
    }
    const types = [];
    if (index === errorOffset) {
      types.push(SpanType.Error);
    }
    if (predictions.has(index)) {
      types.push(SpanType.Predicted);
    }
    if (ranking[0] === index) {
      types.push(SpanType.First);
    }
    if (toReference.has(index)) {
      const ref = React.createRef<ISpanHandles>();
      spans.push(
        <Span
          key={index}
          index={index}
          ref={ref}
          code={code.charAt(index)}
          types={types}
          selected={selectedOffset === index}
        />
      );
      if (index === errorOffset) {
        errorRef = ref;
      }
      const rank = firstPredictions.indexOf(index);
      if (rank !== -1) {
        predictionRefs[rank] = ref;
      }
    } else {
      spans.push(
        <Span
          key={index}
          index={index}
          code={code.charAt(index)}
          types={types}
          selected={selectedOffset === index}
        />
      );
    }
    currentIndex = index + 1;
  }
  if (currentIndex < code.length) {
    spans.push(
      <Span
        key={currentIndex}
        code={code.slice(currentIndex)}
        types={[SpanType.NotPredicted]}
        selected={false}
      />
    );
  }
  return [spans, errorRef!, predictionRefs];
};

const Code = (props: ICodeProps) => {
  const [spans, errorRef, predictionRefs] = useComputeSpans(
    props.code,
    props.ranking,
    props.errorOffset,
    props.selectedOffset
  );
  const predictionButtons = predictionRefs.map((r, i) => (
    <Button
      key={i}
      onClick={() => {
        if (r.current !== null) r.current.scrollIntoView();
      }}
    >
      {i + 1}
    </Button>
  ));
  return (
    <Card className="code">
      <Card.Header>
        <Row>
          <Col sm="auto">
            <TaskPicker
              inputTask={props.inputTask}
              numberOfTasks={props.numberOfTasks}
            />
          </Col>
          <Col />
          <Col sm="auto">
            <ButtonToolbar>
              <ButtonGroup className="mr-2">
                <Button
                  onClick={() => {
                    if (errorRef.current !== null)
                      errorRef.current.scrollIntoView();
                  }}
                >
                  E
                </Button>
              </ButtonGroup>
              <ButtonGroup>{predictionButtons}</ButtonGroup>
            </ButtonToolbar>
          </Col>
        </Row>
      </Card.Header>
      <Card.Body>
        <pre>
          <code>{spans}</code>
        </pre>
      </Card.Body>
    </Card>
  );
};

export default Code;
