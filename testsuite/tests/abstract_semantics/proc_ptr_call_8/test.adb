procedure Test is
   function F return Integer is
   begin
      return 42;
   end F;

   P : access function return Integer := F'Access;
   R : Integer := P.all;
begin
   null;
end Test;
