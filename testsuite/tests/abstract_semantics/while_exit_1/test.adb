procedure Test is
   type MyInt is range -1000 .. 1000;
   x : MyInt := 0;
   y : MyInt := 100;
begin
   while x < y loop
      x := x + 1;
      exit when x = 50;
   end loop;
end Ex1;
