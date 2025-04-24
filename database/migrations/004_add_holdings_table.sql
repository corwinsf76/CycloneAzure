CREATE TABLE IF NOT EXISTS holdings (
    id SERIAL PRIMARY KEY,
    coin_id VARCHAR(50) NOT NULL,
    quantity DECIMAL NOT NULL,
    purchase_price DECIMAL NOT NULL,
    purchase_date TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_coin
        FOREIGN KEY(coin_id)
        REFERENCES coins(id)
);

CREATE INDEX idx_holdings_coin_id ON holdings(coin_id);
