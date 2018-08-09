procedure Test is
   type MyInt is range -1000 .. 1000;
   x : MyInt := 0;
   y : MyInt;
begin
   x := if y > 50 then 1 else -1;
end Ex1;
