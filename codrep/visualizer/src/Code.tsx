import React from "react";
import Span, { ISpanHandles, SpanType } from "./Span";
import Button from "react-bootstrap/Button";
import ButtonGroup from "react-bootstrap/ButtonGroup";
import ButtonToolbar from "react-bootstrap/ButtonToolbar";
import Card from "react-bootstrap/Card";
import Col from "react-bootstrap/Col";
import Row from "react-bootstrap/Row";

export interface IProps {
  code: string;
  ranking: number[];
  error_offset: number;
}

const computeSpans = (
  code: string,
  ranking: number[],
  error_offset: number,
  n_pred_refs = 5
): [
  JSX.Element[],
  React.RefObject<ISpanHandles>,
  React.RefObject<ISpanHandles>[]
] => {
  const spans: JSX.Element[] = [];
  const indices = Array.from(new Set([...ranking, error_offset]));
  const firstPredictions = ranking.slice(0, n_pred_refs);
  const toReference = new Set([error_offset, ...firstPredictions]);
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
        />
      );
    }
    const types = [SpanType.Predicted];
    if (index === error_offset) {
      types.push(SpanType.Error);
    }
    if (ranking[0] === index) {
      types.push(SpanType.First);
    }
    if (toReference.has(index)) {
      const ref = React.createRef<ISpanHandles>();
      spans.push(
        <Span key={index} ref={ref} code={code.charAt(index)} types={types} />
      );
      if (index === error_offset) {
        errorRef = ref;
      }
      const rank = firstPredictions.indexOf(index);
      if (rank !== -1) {
        predictionRefs[rank] = ref;
      }
    } else {
      spans.push(<Span key={index} code={code.charAt(index)} types={types} />);
    }
    currentIndex = index + 1;
  }
  if (currentIndex < code.length) {
    spans.push(
      <Span
        key={currentIndex}
        code={code.slice(currentIndex)}
        types={[SpanType.NotPredicted]}
      />
    );
  }
  return [spans, errorRef!, predictionRefs];
};

const Code = (props: IProps) => {
  const [spans, errorRef, predictionRefs] = computeSpans(
    props.code,
    props.ranking,
    props.error_offset
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
    <Card>
      <Card.Header>
        <Row>
          <Col>Code</Col>
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
