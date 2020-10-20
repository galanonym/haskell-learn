main = do
  line <- getLine
  if null line
    then return () -- (return ()) gives I/O action yielding () that is empty tuple or unit
    else (do -- do block here is one I/O action
      putStrLn $ reverseWords line
      main) -- we run main recursively to read next line

reverseWords :: String -> String
reverseWords = unwords . map reverse . words
