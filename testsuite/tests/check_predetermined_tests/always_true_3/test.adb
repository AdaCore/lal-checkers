procedure Test is
   type Unsigned is mod 10
   y : Unsigned := 0;
begin
   while y >= 0 loop
      y := y + 1;
   end loop;
end Ex1;
