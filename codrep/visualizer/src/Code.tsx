import React, { useMemo } from "react";
import Span, { SpanType } from "./Span";
import Card from "react-bootstrap/Card";

export interface IProps {
  code: string;
  ranking: number[];
  error_offset: number;
}

const computeSpans = (
  code: string,
  ranking: number[],
  error_offset: number
): JSX.Element[] => {
  const spans: JSX.Element[] = [];
  const rankingSet = new Set(ranking);
  const indices = Array.from(new Set([...ranking, error_offset]));
  indices.sort((a, b) => a - b);
  // const ranks = ranking.map((_, index) => index);
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
    spans.push(<Span key={index} code={code.charAt(index)} types={types} />);
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
  return spans;
};

const Code = (props: IProps) => {
  const spans = useMemo(
    () => computeSpans(props.code, props.ranking, props.error_offset),
    [props.code, props.ranking, props.error_offset]
  );
  return (
    <Card>
      <Card.Header>Code</Card.Header>
      <Card.Body>
        <pre>
          <code>{spans}</code>
        </pre>
      </Card.Body>
    </Card>
  );
};

export default Code;
