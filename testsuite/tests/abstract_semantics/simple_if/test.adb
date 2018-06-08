procedure Ex1 is
   type MyInt is range -1000 .. 1000;
   x : MyInt := 0;
   y : MyInt;
begin
   if y > 50 then
      x := 1;
   else
      x := -1;
   end if;
end Ex1;
