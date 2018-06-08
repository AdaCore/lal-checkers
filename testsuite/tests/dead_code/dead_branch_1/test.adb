procedure Ex1 is
   type Integer_Access is access all Integer;
   x : Integer_Access;
   y : Integer;
begin
   if y < 0 then
      y := 40
   end if;

   if y < 0 then
      y := 50;
   end if;
end Ex1;
