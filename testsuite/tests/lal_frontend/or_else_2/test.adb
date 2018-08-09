procedure Test is
   type Integer_Access is access all Integer;
   x : Integer_Access;
   y : Integer := 0;
begin
   if x = null or else x.all = y then
      y := 42;
   end if;
end Ex1;
