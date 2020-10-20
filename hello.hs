import GHC.Unicode

main = do
  putStrLn "Hello, what is your name?"
  name <- getLine
  let bigName = map toUpper name -- we use let to define normal values inside I/O actions
  putStrLn ("Hey " ++ bigName ++ ", you are the boss.")

