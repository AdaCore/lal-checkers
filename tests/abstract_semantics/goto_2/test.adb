procedure Ex1 is
   x : Integer;
   y : Integer;
begin
   while x /= 10 loop
      if x = 5 then
         goto Loop_End;
      end if;
      x := x + 1;
   end loop;

   <<Loop_End>>
   y := x;
end Ex1;
