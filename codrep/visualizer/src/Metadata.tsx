import React from "react";
import Card from "react-bootstrap/Card";
import "./css/Metadata.css";

export interface IMetadataProps {
  columns: string[];
  values: string[];
  rank: number | null;
}

const Metadata = (props: IMetadataProps) => {
  const cards = [];
  if (props.rank !== null) {
    cards.push(
      <Card>
        <Card.Header>Rank</Card.Header>
        <Card.Body>{props.rank}</Card.Body>
      </Card>
    );
  }
  cards.push(
    ...props.values.map((value, i) => {
      const column = props.columns[i];
      return (
        <Card>
          <Card.Header>{column}</Card.Header>
          <Card.Body>{value}</Card.Body>
        </Card>
      );
    })
  );
  return (
    <Card className="metadata">
      <Card.Header>Metadata</Card.Header>
      <Card.Body>{cards}</Card.Body>
    </Card>
  );
};

export default Metadata;
