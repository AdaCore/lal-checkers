procedure Test is
   subtype My_Range is Integer range 0 .. 10;
   S : Integer := 0;
begin
   for I in My_Range loop
      S := S + I;
   end loop;
end Ex1;
